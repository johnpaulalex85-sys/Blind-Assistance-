class IntentRouter:
    def route_intent(self, text: str) -> str:
        text = text.lower()
        if "what" in text and ("see" in text or "around" in text) or "describe" in text or "analy" in text:
            return "SCENE_QUERY"
        elif "who" in text:
            return "PERSON_QUERY"
        elif "read" in text or "text" in text:
            return "TEXT_QUERY"
        elif "far" in text or "distance" in text:
            return "DISTANCE_QUERY"
        elif "danger" in text or "safe" in text or "careful" in text:
            return "SAFETY_QUERY"
        elif "tell me more" in text or "what about" in text:
            return "FOLLOW_UP_QUERY"
        else:
            return "GENERAL_QUESTION"
