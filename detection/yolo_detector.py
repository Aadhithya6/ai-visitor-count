from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov8n-face.pt', conf_threshold=0.5):
        """
        Initialize YOLOv8 face detector.
        If the model file doesn't exist, it will be downloaded (if available in ultralytics) 
        or you should provide the correct path.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Detect faces in a frame.
        Returns a list of bounding boxes [x1, y1, x2, y2, confidence].
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Support yolov8n.pt general model by only tracking persons (class 0)
                cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') and box.cls is not None else 0
                if cls_id != 0:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])
                
        return detections
