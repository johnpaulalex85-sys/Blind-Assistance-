import logging

logger = logging.getLogger(__name__)

def _is_inside(inner_box, outer_box):
    if not inner_box or not outer_box or len(inner_box) != 4 or len(outer_box) != 4:
        return False
    xA = max(inner_box[0], outer_box[0])
    yA = max(inner_box[1], outer_box[1])
    xB = min(inner_box[2], outer_box[2])
    yB = min(inner_box[3], outer_box[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    innerArea = max(0, inner_box[2] - inner_box[0]) * max(0, inner_box[3] - inner_box[1])
    if innerArea == 0:
        return False
    return (interArea / innerArea) > 0.5

def _get_distance_for_box(box, distances):
    if not box:
        return ""
    for d in distances:
        if d.get("box") == box:
            return d.get("distance", "")
    return ""

class DescribeService:
    def __init__(self):
        logger.info("Initializing Rule-Based Describe Service (0 MB RAM)")
        
    def generate_description(self, frame, scene_json: dict) -> str:
        """
        Generates a fast, rule-based description using the structured scene data.
        Bypasses heavy LLMs entirely.
        """
        try:
            faces = scene_json.get("faces", [])
            objects = scene_json.get("objects", [])
            texts = scene_json.get("texts", [])
            distances = scene_json.get("distances", [])
            
            # Filter out "person" objects that contain a detected face
            filtered_objects = []
            for obj in objects:
                if obj.get("class_name") == "person":
                    has_face = False
                    for face in faces:
                        if _is_inside(face.get("box"), obj.get("box")):
                            has_face = True
                            break
                    if has_face:
                        continue
                filtered_objects.append(obj)
            objects = filtered_objects
            
            parts = []
            
            # Describe faces
            if faces:
                known_faces = []
                for f in faces:
                    if f["name"].lower() != "unknown" and not f["name"].startswith("Person "):
                        dist = _get_distance_for_box(f.get("box"), distances)
                        name_str = f["name"]
                        if dist:
                            name_str += f" at {dist}"
                        known_faces.append(name_str)
                        
                unknown_faces = len([f for f in faces if f["name"].lower() == "unknown" or f["name"].startswith("Person ")])
                
                face_parts = []
                if known_faces:
                    face_parts.append(f"{', '.join(known_faces)}")
                if unknown_faces > 0:
                    face_parts.append(f"{unknown_faces} unrecognized person{'s' if unknown_faces > 1 else ''}")
                    
                if face_parts:
                    parts.append(f"I see {' and '.join(face_parts)}.")

            # Describe objects and distances
            if objects:
                # Group objects by class name and find their closest distance
                obj_summary = {}
                for obj in objects:
                    cls = obj.get("class_name")
                    dist_str = _get_distance_for_box(obj.get("box"), distances)
                    if cls not in obj_summary:
                        obj_summary[cls] = []
                    if dist_str:
                        obj_summary[cls].append(dist_str)
                
                # Format object strings
                obj_strs = []
                # Only take the top 3 most relevant objects to avoid overwhelming the user
                for cls, dist_list in list(obj_summary.items())[:3]:
                    if dist_list:
                        obj_strs.append(f"a {cls} at {dist_list[0]}")
                    else:
                        obj_strs.append(f"a {cls}")
                        
                if obj_strs:
                    parts.append(f"Nearby, there is {', '.join(obj_strs)}.")
                    
            # Describe text
            if texts:
                # Sort text by vertical position (y1) then horizontal (x1) to approximate reading order
                # We group y-coordinates by chunks (e.g., 10 pixels) to handle slight misalignments on the same line
                sorted_texts = sorted(texts, key=lambda t: (t.get("box", [0,0,0,0])[1] // 15, t.get("box", [0,0,0,0])[0]))
                
                valid_texts = []
                for t in sorted_texts:
                    clean_text = t.get("text", "").strip()
                    if clean_text:
                        valid_texts.append(clean_text)
                
                if valid_texts:
                    combined_text = " ".join(valid_texts)
                    parts.append(f"I can read the text: '{combined_text}'.")
                        
            if not parts:
                return "I don't see anything clearly at the moment."
                
            return " ".join(parts)
            
        except Exception as e:
            logger.error(f"Rule-based description error: {e}")
            return "Error generating scene description."
