import numpy as np
import insightface
from insightface.app import FaceAnalysis
import cv2

class InsightFaceModel:
    def __init__(self, model_pack='buffalo_l', ctx_id=0):
        """
        Initialize InsightFace model.
        model_pack: 'buffalo_l' is a common large model.
        ctx_id: 0 for GPU if available, -1 for CPU.
        """
        # Note: In a production environment, you might want to specify and download the model manually.
        # FaceAnalysis will attempt to download it to ~/.insightface/models/
        self.app = FaceAnalysis(name=model_pack, providers=['CPUExecutionProvider'])
        # If GPU is needed, change providers to ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def get_embedding(self, face_image):
        """
        Produce a normalized 512-dimension embedding for the face image.
        Note: FaceAnalysis expects the full image or a crop. 
        InsightFace's app.get() performs its own detection and alignment.
        """
        # face_image should be a BGR image crop or full frame
        faces = self.app.get(face_image)
        
        if len(faces) == 0:
            return None
            
        # For simplicity, we assume the first face is the intended one if multiple are in crop.
        # In a real pipeline, we'd pass the crop from YOLO.
        # However, InsightFace likes to do its own detection/alignment for embeddings.
        embedding = faces[0].normed_embedding
        return embedding

    def generate_embedding_for_crop(self, crop):
        """
        Generate embedding for a pre-cropped face (from YOLO).
        """
        faces = self.app.get(crop)
        if len(faces) == 0:
            return None
        return faces[0].normed_embedding

    def detect_and_embed(self, frame):
        """
        Run the InsightFace inference on the full frame.
        Extract bounding boxes, confidence scores, and normalized embeddings.
        Returns a list of dicts:
        [{'bbox': [x1, y1, x2, y2], 'det_score': float, 'embedding': np.array}]
        """
        faces = self.app.get(frame)
        results = []
        for face in faces:
            # face.bbox is usually [x1, y1, x2, y2]
            # face.det_score is confidence float
            # face.normed_embedding is a 512-dim L2-normalized array
            results.append({
                'bbox': face.bbox,
                'det_score': face.det_score,
                'embedding': face.normed_embedding
            })
        return results
