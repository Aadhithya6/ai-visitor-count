from deep_sort_realtime.deepsort_tracker import DeepSort

class FaceTracker:
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2):
        """
        Initialize DeepSORT tracker.
        max_age: Number of frames to keep a track alive without update.
        n_init: Number of frames before a track is confirmed.
        nms_max_overlap: Maximum overlap for NMS.
        max_cosine_distance: Maximum distance for cosine similarity (internal DeepSORT tracking).
        """
        self.tracker = DeepSort(
            max_age=max_age, 
            n_init=n_init, 
            nms_max_overlap=nms_max_overlap, 
            max_cosine_distance=max_cosine_distance
        )

    def update(self, detections, frame):
        """
        Update tracker with detections.
        detections: List of [x1, y1, x2, y2, confidence, class_id (optional)]
        Returns a list of track objects.
        """
        # detections format for deep_sort_realtime: [ ([left, top, w, h], confidence, label) ]
        formatted_detections = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w, h = x2 - x1, y2 - y1
            formatted_detections.append(([x1, y1, w, h], conf, "face"))
            
        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
        return tracks
