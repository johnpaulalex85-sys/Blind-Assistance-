import logging
import asyncio
import numpy as np
import time
from .tracker import Sort
from services import get_yolo_detector, get_ocr_service, get_face_service, get_depth_service
from config import settings

logger = logging.getLogger(__name__)

class PerceptionManager:
    def __init__(self):
        self.tracker = Sort(
            max_age=settings.TRACKER_MAX_AGE,
            min_hits=settings.TRACKER_MIN_HITS,
            iou_threshold=settings.TRACKER_IOU_THRESHOLD
        )
        self.yolo_detector = get_yolo_detector()
        self.ocr_service = get_ocr_service()
        self.face_service = get_face_service()
        self.depth_service = get_depth_service()
        
        self.frame_count = 0
        self.cached_ocr = {"texts": []}
        self.cached_face = {"faces": []}
        self.cached_depth = ([], None)
        logger.info("Perception Manager initialized.")

    async def analyze_frame(self, frame: np.ndarray):
        """
        Runs full perception on a frame.
        Returns a unified observation dictionary.
        """
        if frame is None:
            return None
            
        self.frame_count += 1

        # YOLO runs every frame
        yolo_task = asyncio.to_thread(self.yolo_detector.detect, frame, settings.YOLO_CONF_THRESHOLD)
        
        tasks = [yolo_task]
        
        run_face_this_frame = (self.frame_count % 5 == 0)
        run_ocr_this_frame = (self.frame_count % 10 == 0)
        
        async def run_ocr():
            if self.ocr_service is None: return {"texts": []}
            return await asyncio.to_thread(self.ocr_service.detect_text, frame, settings.OCR_CONF_THRESHOLD)
            
        async def run_face():
            if self.face_service is None: return {"faces": []}
            return await asyncio.to_thread(self.face_service.recognize_face, frame, settings.FACE_CONF_THRESHOLD)

        if run_ocr_this_frame: tasks.append(run_ocr())
        if run_face_this_frame: tasks.append(run_face())
        
        results = await asyncio.gather(*tasks)
        yolo_dets = results[0]
        
        idx = 1
        if run_ocr_this_frame:
            self.cached_ocr = results[idx]
            idx += 1
        if run_face_this_frame:
            self.cached_face = results[idx]
            
        ocr_data = self.cached_ocr
        face_data = self.cached_face

        # 2. Track YOLO objects
        tracker_input = []
        det_class_mapping = []
        for det in yolo_dets:
            box = det["box"]
            tracker_input.append([box[0], box[1], box[2], box[3], det["confidence"]])
            det_class_mapping.append(det["class_name"])
            
        if tracker_input:
            tracker_input = np.array(tracker_input)
        else:
            tracker_input = np.empty((0, 5))
            
        tracked_objects_raw = self.tracker.update(tracker_input)
        
        tracked_objects = []
        for trk in tracked_objects_raw:
            x1, y1, x2, y2, obj_id = trk
            
            cx, cy = (x1+x2)/2, (y1+y2)/2
            best_class = "unknown"
            min_dist = float('inf')
            
            for d, cls_name in zip(tracker_input, det_class_mapping):
                dx, dy = (d[0]+d[2])/2, (d[1]+d[3])/2
                dist = (cx-dx)**2 + (cy-dy)**2
                if dist < min_dist:
                    min_dist = dist
                    best_class = cls_name
                    
            tracked_objects.append({
                "track_id": int(obj_id),
                "class_name": best_class,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": 1.0 
            })

        # 3. Add OCR and Faces to the final detection list for Depth
        all_detections_for_depth = list(tracked_objects)
        
        for item in ocr_data.get("texts", []):
            bbox = item["bbox"]
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            all_detections_for_depth.append({
                "box": [min(xs), min(ys), max(xs), max(ys)],
                "confidence": item["confidence"],
                "class_name": f'text: "{item["text"]}"'
            })
            
        for item in face_data.get("faces", []):
            all_detections_for_depth.append({
                "box": item["box"],
                "confidence": item["confidence"],
                "class_name": f'face: {item["name"]}'
            })

        # 4. Run Depth (every 5 frames, staggered from face to reduce peak CPU load)
        run_depth_this_frame = (self.frame_count % 5 == 2)
        if run_depth_this_frame and self.depth_service is not None:
            self.cached_depth = await asyncio.to_thread(self.depth_service.estimate_depth, frame, all_detections_for_depth)
            
        distance_data, depth_map = self.cached_depth

        # 5. Build unified observation
        observation = {
            "timestamp": time.time(),
            "objects": tracked_objects,
            "faces": face_data.get("faces", []),
            "texts": ocr_data.get("texts", []),
            "distances": distance_data,
            "depth_map_base64": depth_map,
            "raw_yolo": yolo_dets
        }

        return observation

# Global instance
_perception_manager = None
def get_perception_manager():
    global _perception_manager
    if _perception_manager is None:
        _perception_manager = PerceptionManager()
    return _perception_manager
