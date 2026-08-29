import logging
from ultralytics import YOLO
import numpy as np
import threading
import time

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        logger.info(f"Loading YOLO model {model_name}...")
        try:
            self.model = YOLO(model_name)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
            
        self.latest_detections = []
        self.running = False
        self.thread = None
        self.camera_manager = None

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.4):
        if self.model is None:
            return []
            
        try:
            results = self.model.predict(source=frame, conf=conf_threshold, verbose=False)
        except Exception as e:
            logger.error(f"YOLO predict error: {e}")
            return []

        detections = []
        if len(results) > 0:
            result = results[0]
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = result.names[cls_id]
                
                detections.append({
                    "box": [round(x, 2) for x in xyxy],
                    "confidence": round(conf, 3),
                    "class_name": cls_name
                })
                
        return detections

    def start_live_loop(self, camera_manager, conf_threshold=0.4):
        """Continuously process frames in a background thread."""
        if self.running: return
        self.camera_manager = camera_manager
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(conf_threshold,), daemon=True)
        self.thread.start()
        logger.info("Live YOLO detection loop started.")

    def _loop(self, conf_threshold):
        while self.running:
            if self.camera_manager:
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    # Run detection and store the results instantly
                    dets = self.detect(frame, conf_threshold)
                    self.latest_detections = dets
            # Run at roughly ~20 FPS max to conserve CPU
            time.sleep(0.05) 

    def stop_live_loop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Live YOLO detection loop stopped.")
