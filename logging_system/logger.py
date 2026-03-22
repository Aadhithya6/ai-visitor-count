import os
import logging
from datetime import datetime

def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "events.log")
    
    # Configure logging
    logger = logging.getLogger("FaceTracker")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_event(logger, face_id, event_type):
    message = f"FACE_ID={face_id} EVENT={event_type.upper()}"
    logger.info(message)

def get_image_save_path(log_dir, event_type):
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(log_dir, f"{event_type}s", date_str)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir
