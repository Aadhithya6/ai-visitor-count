import cv2
import numpy as np
import os
from datetime import datetime

def crop_face(frame, bbox, min_size=(30, 30)):
    """
    Crop region from frame given bounding box [x1, y1, x2, y2]
    without resizing, preserving original quality for the embedder.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None

    # Enforce minimum size to prevent poor quality embeddings
    if (x2 - x1) < min_size[0] or (y2 - y1) < min_size[1]:
        return None
        
    crop = frame[y1:y2, x1:x2]
    return crop

def is_blurry(image, threshold=100.0):
    """
    Compute the Laplacian variance to detect blur.
    Returns True if the image is blurry (variance < threshold).
    """
    if image is None or image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def normalize_embedding(embedding):
    """
    Apply L2 normalization to the embedding array.
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def compute_cosine_similarity(a, b):
    """
    Compute cosine similarity between two normalized embeddings.
    Since they are normalized, the dot product is equivalent to geometric cosine similarity.
    """
    return np.dot(a, b)

def save_cropped_face(crop, save_dir, face_id):
    """
    Save cropped face image to disk.
    """
    timestamp = datetime.now().strftime("%H-%M-%S-%f")
    filename = f"{face_id}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, crop)
    return filepath
