from config import settings

class SafetyEngine:
    def evaluate_safety(self, state):
        """
        Returns (is_hazard, priority, message)
        """
        for obj in state.objects:
            if obj.distance_val > 0 and obj.distance_val < settings.SAFETY_CRITICAL_DISTANCE:
                return True, "CRITICAL", f"Careful. Obstacle directly ahead: {obj.class_name}."
            
            # Special hazard classes
            hazard_classes = ["car", "truck", "bus", "train", "stairs", "fire hydrant", "stop sign"]
            if obj.class_name in hazard_classes and obj.distance_val > 0 and obj.distance_val < 3.0:
                return True, "HIGH", f"Hazard detected: {obj.class_name} ahead."

        return False, "NORMAL", ""
