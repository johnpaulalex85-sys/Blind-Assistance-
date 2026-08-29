import datetime
import json

def build_scene_json(yolo_dets, ocr_dets, face_dets, distances):
    """
    Merges outputs from various models into a single structured JSON object.
    
    Args:
        yolo_dets: List of YOLO detections [{"class_name": ..., "confidence": ..., "box": [...]}]
        ocr_dets: Dict from OCRService {"texts": [{"text": ..., "confidence": ..., "bbox": [...]}]}
        face_dets: Dict from FaceService {"faces": [{"name": ..., "confidence": ..., "box": [...]}]}
        distances: List from DepthService [{"object": ..., "distance": ..., "box": [...]}]
        
    Returns:
        A dictionary representing the unified scene.
    """
    
    # Process YOLO objects (filter out ones that are just OCR or Face duplicates if any, though our pipeline keeps them separate before merging in app.py typically. We will assume yolo_dets are purely from YOLO).
    objects = []
    if yolo_dets:
        for det in yolo_dets:
            # Check if this is an injected face or OCR from app.py, if so skip it to avoid duplication if we are passed raw YOLO dets.
            if det.get("class_name", "").startswith('"') or det.get("class_name", "").startswith("Face: "):
                continue
            objects.append({
                "class_name": det.get("class_name"),
                "confidence": det.get("confidence"),
                "box": det.get("box")
            })

    # Process faces
    faces = []
    if face_dets and "faces" in face_dets:
        for face in face_dets["faces"]:
            faces.append({
                "name": face.get("name"),
                "confidence": face.get("confidence"),
                "box": face.get("box")
            })

    # Process text
    texts = []
    if ocr_dets and "texts" in ocr_dets:
        for text_obj in ocr_dets["texts"]:
            # OCR bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]. Convert to [x1, y1, x2, y2]
            bbox = text_obj.get("bbox", [])
            if len(bbox) == 4:
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                box = [min(xs), min(ys), max(xs), max(ys)]
            else:
                box = []
                
            texts.append({
                "text": text_obj.get("text"),
                "confidence": text_obj.get("confidence"),
                "box": box
            })

    scene_data = {
        "objects": objects,
        "faces": faces,
        "texts": texts,
        "distances": distances if distances else [],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    return scene_data
