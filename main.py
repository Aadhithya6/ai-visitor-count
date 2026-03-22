import cv2
import json
import os
import sys
import numpy as np
from datetime import datetime
import threading
from collections import deque

from detection.yolo_detector import YOLODetector
from recognition.insightface_model import InsightFaceModel
from tracking.tracker import FaceTracker
from database.db import DatabaseManager, Face, FaceEmbedding, Event, EventType
from logging_system.logger import setup_logger, log_event, get_image_save_path
from utils.helpers import crop_face, save_cropped_face, compute_cosine_similarity, is_blurry, normalize_embedding

class FaceTrackingApp:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.logger = setup_logger(self.config['log_dir'])
        self.db = DatabaseManager(self.config['db_url'])
        
        # Defer heavy model loading — these are loaded in _load_models()
        self.detector = None
        self.recognizer = None
        self.tracker = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Mapping: track_id -> face_id (UUID)
        self.track_to_face = {}
        # Mapping: face_id -> last_seen_frame_count
        self.active_faces = {}
        # Track Buffer for delaying registration: track_id -> list of valid embeddings
        self.track_buffer = {}
        # Cache for recently departed faces to prevent instant re-logging duplicates
        self.recently_exited = {}
        
        self.new_faces_in_session = set()
        
        self.frame_count = 0
        self.visitor_count = self.get_total_unique_visitors()
        
        # Cross-thread shared state for API access
        self.current_frame = None
        self.lock = threading.Lock()
        self.recent_logs = deque(maxlen=20)
        self.show_gui = True

    def _load_models(self):
        """Load heavy AI models AFTER the stream is opened to prevent timeouts."""
        self.logger.info("Loading AI models...")
        self.detector = YOLODetector(self.config['yolo_model'])
        self.recognizer = InsightFaceModel(self.config['insightface_model'])
        self.tracker = FaceTracker(max_age=self.config['exit_frame_threshold'])
        self.logger.info("All models loaded successfully.")

    def get_total_unique_visitors(self):
        session = self.db.get_session()
        count = session.query(Face).count()
        session.close()
        return count

    def find_match_in_db(self, embedding_list):
        """
        Compare a list of buffered embeddings against all stored 
        embeddings in the FaceEmbedding table. Return the best matching face_id.
        """
        session = self.db.get_session()
        faces = session.query(Face).all()
        
        best_match = None
        best_sim = -1
        
        # Generate an average embedding from the buffer to make matching more robust against noise
        if len(embedding_list) > 1:
            avg_emb = np.mean(embedding_list, axis=0)
            target_embedding = normalize_embedding(avg_emb)
        else:
            target_embedding = embedding_list[0]
            
        for face in faces:
            # Check maximum similarity across all stored variations of this person's face
            for db_emb in face.embeddings:
                stored_embedding = np.frombuffer(db_emb.embedding, dtype=np.float32)
                sim = compute_cosine_similarity(target_embedding, stored_embedding)
                if sim > self.config['similarity_threshold'] and sim > best_sim:
                    best_sim = sim
                    best_match = face.face_id
        
        if best_match:
            self.logger.info(f"MATCH FOUND: face_id={best_match} | max_sim={best_sim:.4f}")
        else:
            self.logger.info(f"NO MATCH: max_sim={best_sim:.4f} (threshold={self.config['similarity_threshold']})")
            
        session.close()
        return best_match

    def register_or_update_face(self, target_embeddings, face_id=None):
        """
        Register a new face_id or append embeddings to an existing face_id.
        """
        session = self.db.get_session()
        
        if face_id is None:
            new_face = Face()
            session.add(new_face)
            session.commit()
            face_id = new_face.face_id
            with self.lock:
                self.visitor_count += 1
            self.logger.info(f"REGISTERED NEW FACE: face_id={face_id}")
            self.new_faces_in_session.add(face_id)
        
        # Save up to 3 best embeddings from the buffer to the DB to improve future recall
        for emb in target_embeddings[:3]:
            new_emb = FaceEmbedding(face_id=face_id, embedding=emb.tobytes())
            session.add(new_emb)
        session.commit()
        session.close()
        return face_id

    def add_log_entry(self, face_id, event_type):
        """Helper to add log to recent_logs for API"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] Face {face_id[:8]} {event_type}"
        with self.lock:
            self.recent_logs.appendleft(log_entry)

    def log_db_event(self, face_id, event_type, image_path):
        session = self.db.get_session()
        event = Event(face_id=face_id, event_type=event_type, image_path=image_path)
        session.add(event)
        session.commit()
        session.close()
        
    def _is_recently_exited(self, face_id):
        # Prevent double entry-logs if they just 'exited' a few frames ago due to obstruction
        if face_id in self.recently_exited:
            if self.frame_count - self.recently_exited[face_id] < (self.config['exit_frame_threshold'] * 2):
                return True
        return False

    def run(self):
        video_source = self.config['video_source']
        if str(video_source).isdigit():
            video_source = int(video_source)
            
        # Optimize for live streams like RTSP or Webcams to prevent severe lag
        is_live = isinstance(video_source, int) or str(video_source).startswith(("rtsp://", "http://"))
        if is_live:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|stimeout;60000000"
            
        cap = cv2.VideoCapture(video_source)
        
        # Reduce OpenCV buffer to 1 frame for live streams to avoid processing stale frames
        if is_live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.logger.error(f"Cannot open video source: {video_source}")
            return
        
        self.logger.info(f"Stream opened: {video_source}")
        
        # Read and discard a warm-up frame to keep the stream alive
        ret, _ = cap.read()
        if not ret:
            self.logger.error("Failed to read initial frame from stream.")
            return
        
        # NOW load the heavy AI models (stream is already open and buffering)
        self._load_models()

        # Calculate smaller window dimensions (50% scale)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.show_gui:
            cv2.namedWindow("Intelligent Face Tracker", cv2.WINDOW_NORMAL)
            scaled_w = int(width * 0.5) if width > 0 else 800
            scaled_h = int(height * 0.5) if height > 0 else 600
            cv2.resizeWindow("Intelligent Face Tracker", scaled_w, scaled_h)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Skip frames logic
            if self.frame_count % self.config['detection_skip_frames'] != 0:
                pass 
            
            # 2. Detection using YOLO (Fast, detects bodies/faces from far away)
            detections = self.detector.detect(frame)
                
            # 3. Tracking Update
            tracks = self.tracker.update(detections, frame)
            
            current_frame_faces = set()
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                bbox = track.to_ltrb() # [x1, y1, x2, y2]
                
                # Check if we already have a face_id for this track
                if track_id not in self.track_to_face:
                    # Multi-frame validation: Ensure we capture high-quality crops
                    crop = crop_face(frame, bbox)
                    if crop is not None and not is_blurry(crop, self.config['blur_threshold']):
                        
                        # FAST PRE-FILTER: Only run heavy InsightFace if Haar detects a face in the crop
                        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        fast_faces = self.face_cascade.detectMultiScale(gray_crop, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        if len(fast_faces) > 0:
                            target_emb = self.recognizer.generate_embedding_for_crop(crop)
                                    
                            if target_emb is not None:
                                if track_id not in self.track_buffer:
                                    self.track_buffer[track_id] = []
                                self.track_buffer[track_id].append(target_emb)
                                
                                # Do we have enough frames to confidently register/match?
                                if len(self.track_buffer[track_id]) >= self.config['min_frames_for_registration']:
                                    # Match in DB
                                    face_id = self.find_match_in_db(self.track_buffer[track_id])
                                    
                                    if face_id is None:
                                        face_id = self.register_or_update_face(self.track_buffer[track_id])
                                    
                                    self.track_to_face[track_id] = face_id
                                    
                                    # Log ENTRY if not active and not recently exited
                                    if face_id not in self.active_faces and not self._is_recently_exited(face_id):
                                        save_dir = get_image_save_path(self.config['log_dir'], "entry")
                                        filepath = save_cropped_face(crop, save_dir, face_id)
                                        self.log_db_event(face_id, EventType.ENTRY, filepath)
                                        log_event(self.logger, face_id, "ENTRY")
                                        self.add_log_entry(face_id, "ENTERED")
                                    self.active_faces[face_id] = self.frame_count
                                    
                                    # Clean up buffer to save memory
                                    del self.track_buffer[track_id]
                else:
                    # We already know who this track is
                    face_id = self.track_to_face[track_id]
                    
                    if face_id not in self.active_faces and not self._is_recently_exited(face_id):
                        crop = crop_face(frame, bbox)
                        if crop is not None:
                            save_dir = get_image_save_path(self.config['log_dir'], "entry")
                            filepath = save_cropped_face(crop, save_dir, face_id)
                            self.log_db_event(face_id, EventType.ENTRY, filepath)
                            log_event(self.logger, face_id, "ENTRY RE-ACQUIRED")
                            self.add_log_entry(face_id, "RE-ENTERED")
                    
                    self.active_faces[face_id] = self.frame_count
                
                face_id = self.track_to_face.get(track_id)
                if face_id:
                    current_frame_faces.add(face_id)
                    self.active_faces[face_id] = self.frame_count
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Extract the person crop to find their face precisely
                    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    face_drawn = False
                    
                    if crop.size > 0:
                        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray_crop, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        box_color = (0, 0, 255) if face_id in self.new_faces_in_session else (255, 0, 0)
                        
                        if len(faces) > 0:
                            # Use the largest face found in this crop
                            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                            fx, fy, fw, fh = faces[0]
                            # Translate back to full frame coordinates
                            fx += max(0, x1)
                            fy += max(0, y1)
                            
                            # Draw box exactly around the face
                            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)
                            cv2.putText(frame, f"ID: {face_id[:8]}", (fx, fy - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            face_drawn = True
                    
                    if not face_drawn:
                        box_color = (0, 0, 255) if face_id in self.new_faces_in_session else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, f"ID: {face_id[:8]}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Detect EXIT events
            inactive_faces = []
            for face_id, last_seen in self.active_faces.items():
                if self.frame_count - last_seen > self.config['exit_frame_threshold']:
                    inactive_faces.append(face_id)
            
            for face_id in inactive_faces:
                save_dir = get_image_save_path(self.config['log_dir'], "exit")
                self.log_db_event(face_id, EventType.EXIT, "N/A")
                log_event(self.logger, face_id, "EXIT")
                self.add_log_entry(face_id, "EXITED")
                del self.active_faces[face_id]
                self.recently_exited[face_id] = self.frame_count

            # Prune old recently_exited records
            expired_exits = [fid for fid, frame_num in self.recently_exited.items() 
                             if self.frame_count - frame_num > (self.config['exit_frame_threshold'] * 5)]
            for fid in expired_exits:
                del self.recently_exited[fid]

            # Display Visitor Count
            cv2.putText(frame, f"Unique Visitors: {self.visitor_count}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Store frame for API access
            with self.lock:
                self.current_frame = frame.copy()
            
            if self.show_gui:
                cv2.imshow("Intelligent Face Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Application shut down.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--source', type=str, help='Override video source (e.g., 0 for webcam, rtsp://... for stream)')
    args = parser.parse_args()
    
    app = FaceTrackingApp(config_path=args.config)
    
    # Override video source if provided via CLI
    if args.source is not None:
        app.config['video_source'] = args.source
        
    app.run()
