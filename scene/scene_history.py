from .scene_state import SceneState
import logging

logger = logging.getLogger(__name__)

class SceneChangeDetector:
    def __init__(self):
        self.last_state = None

    def detect_changes(self, current_state: SceneState):
        """
        Compares current_state with last_state.
        Returns a dictionary of events (new_objects, disappeared_objects, approaching_objects, new_faces, new_texts)
        """
        events = {
            "new_objects": [],
            "disappeared_objects": [],
            "approaching_objects": [],
            "new_faces": [],
            "new_texts": []
        }

        if self.last_state is None:
            self.last_state = current_state
            return events

        # Compare objects by track_id
        last_obj_map = {obj.track_id: obj for obj in self.last_state.objects}
        curr_obj_map = {obj.track_id: obj for obj in current_state.objects}

        for track_id, curr_obj in curr_obj_map.items():
            if track_id not in last_obj_map:
                events["new_objects"].append(curr_obj)
            else:
                # Check if getting significantly closer
                last_obj = last_obj_map[track_id]
                # Lower value means closer in this distance system
                if last_obj.distance_val > 0 and curr_obj.distance_val > 0:
                    if curr_obj.distance_val < last_obj.distance_val - 0.5: # 0.5 pseudo-meters threshold
                        events["approaching_objects"].append(curr_obj)

        for track_id, last_obj in last_obj_map.items():
            if track_id not in curr_obj_map:
                events["disappeared_objects"].append(last_obj)

        # Compare faces by name
        last_faces = set([f.name for f in self.last_state.faces if f.name != "unknown person"])
        curr_faces = set([f.name for f in current_state.faces if f.name != "unknown person"])
        new_faces = curr_faces - last_faces
        for f in current_state.faces:
            if f.name in new_faces:
                events["new_faces"].append(f)

        # Compare texts
        last_texts = set([t.text for t in self.last_state.texts])
        curr_texts = set([t.text for t in current_state.texts])
        new_texts = curr_texts - last_texts
        for t in current_state.texts:
            if t.text in new_texts:
                events["new_texts"].append(t)

        self.last_state = current_state
        return events
