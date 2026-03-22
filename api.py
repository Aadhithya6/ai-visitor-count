import os
import cv2
import time
import threading
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from main import FaceTrackingApp

app = FastAPI(title="Visitor Monitoring System API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tracker instance
tracker_app = FaceTrackingApp()

def run_tracker():
    """Background thread to run the face tracking logic"""
    # Disable OpenCV window for server-side processing
    tracker_app.show_gui = False
    # Start the tracker in a separate thread
    tracker_app.run()

@app.on_event("startup")
async def startup_event():
    # Start the tracker in a separate thread
    thread = threading.Thread(target=run_tracker, daemon=True)
    thread.start()

def gen_frames():
    """Video streaming generator"""
    while True:
        with tracker_app.lock:
            if tracker_app.current_frame is None:
                time.sleep(0.1)
                continue
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', tracker_app.current_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Control streaming rate (~20 FPS)
        time.sleep(0.05)

@app.get("/video")
async def video_feed():
    """Video streaming route"""
    return StreamingResponse(gen_frames(), 
                            media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/visitors/count")
async def get_visitor_count():
    """Return current unique visitor count"""
    with tracker_app.lock:
        return {"count": tracker_app.visitor_count}

@app.get("/events")
async def get_events():
    """Return recent entry/exit logs"""
    with tracker_app.lock:
        # Convert deque to list for JSON response
        return {"logs": list(tracker_app.recent_logs)}

@app.get("/status")
async def get_status():
    """Return system status"""
    return {"status": "Running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
