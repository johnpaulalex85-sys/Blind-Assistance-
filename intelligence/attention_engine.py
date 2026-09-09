class AttentionEngine:
    """
    Decides the priority of events detected by SceneChangeDetector.
    """
    def evaluate_priority(self, events: dict):
        priority = "BACKGROUND"
        messages = []

        if events["approaching_objects"]:
            priority = "HIGH"
            # Just take the closest one
            obj = events["approaching_objects"][0]
            messages.append(f"Object approaching: {obj.class_name}.")

        if events["new_faces"]:
            if priority in ["BACKGROUND", "LOW"]:
                priority = "MEDIUM"
            
            announced_objects = [o.class_name for o in events["approaching_objects"]] + [o.class_name for o in events.get("new_objects", [])]
            for face in events["new_faces"]:
                if face.name not in announced_objects:
                    messages.append(f"Person nearby: {face.name}.")

        if events["new_objects"]:
            if priority == "BACKGROUND":
                priority = "LOW"

        # Define an ordering mapping for comparison if needed
        # Priority mapping: CRITICAL > HIGH > MEDIUM > NORMAL > LOW > BACKGROUND
        
        return priority, " ".join(messages)
