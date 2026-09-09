from services import get_describe_service

class VisionTools:
    def __init__(self, memory):
        self.memory = memory

    def get_current_scene(self):
        return self.memory.get_latest_state()

    def describe_scene_complex(self, frame, scene_json) -> str:
        describe_svc = get_describe_service()
        if describe_svc:
            return describe_svc.generate_description(frame, scene_json)
        return "Complex reasoning is currently unavailable."
        
    def answer_question(self, question: str, frame, scene_json: dict) -> str:
        from services import get_gemini_service
        gemini_svc = get_gemini_service()
        if gemini_svc:
            return gemini_svc.answer_question(question, frame, scene_json)
        return "My cloud brain is not currently configured or enabled."
