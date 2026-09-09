import logging
from config import settings
from scene.scene_state import SceneState
from scene.scene_history import SceneChangeDetector
from intelligence.memory import SceneMemory
from intelligence.context_manager import ContextManager
from intelligence.attention_engine import AttentionEngine
from intelligence.safety_engine import SafetyEngine
from intelligence.intent_router import IntentRouter
from intelligence.response_manager import ResponseManager
from tools.vision_tools import VisionTools
from tools.memory_tools import MemoryTools
from tools.audio_tools import AudioTools

logger = logging.getLogger(__name__)

class AssistantBrain:
    def __init__(self):
        self.scene_memory = SceneMemory()
        self.context_manager = ContextManager()
        self.change_detector = SceneChangeDetector()
        
        self.attention = AttentionEngine()
        self.safety = SafetyEngine()
        self.router = IntentRouter()
        
        self.vision_tools = VisionTools(self.scene_memory)
        self.memory_tools = MemoryTools(self.scene_memory, self.context_manager)
        self.audio_tools = AudioTools()
        
        self.response = ResponseManager(self.audio_tools)
        
    def process_new_scene(self, state: SceneState):
        """
        Called continuously when a new scene state is produced by PerceptionManager.
        Handles proactive assistance and safety.
        """
        self.scene_memory.add_state(state)
        
        # 1. Safety Check
        if settings.ASSISTANT_MODE != "QUIET":
            is_hazard, priority, msg = self.safety.evaluate_safety(state)
            if is_hazard:
                self.response.format_and_speak(msg, priority)
                return

        # 2. Scene Change Detection
        if settings.ASSISTANT_MODE == "NORMAL" or settings.ASSISTANT_MODE == "FOCUS":
            events = self.change_detector.detect_changes(state)
            priority, msg = self.attention.evaluate_priority(events)
            if priority not in ["BACKGROUND", "LOW"] and msg:
                self.response.format_and_speak(msg, priority)

    def process_user_query(self, text: str):
        """
        Called when user speaks to the assistant.
        """
        # Interrupt any ongoing background speech
        self.audio_tools.interrupt()
        
        # Mute new background chatter while we handle this
        self.response.mute_background = True
        
        try:
            resolved_text = self.context_manager.resolve_pronoun(text)
            intent = self.router.route_intent(resolved_text)
            
            logger.info(f"User Query: '{text}' -> Resolved: '{resolved_text}' -> Intent: {intent}")
            
            state = self.scene_memory.get_latest_state()
            
            is_async_gemini = False
        
            if intent == "SCENE_QUERY":
                if state and state.objects:
                    objs = list(set([o.class_name for o in state.objects]))[:3]
                    msg = f"I see {', '.join(objs)}."
                    self.response.format_and_speak(msg, "HIGH")
                else:
                    self.response.format_and_speak("I don't see anything clearly right now.", "HIGH")
                    
            elif intent == "PERSON_QUERY":
                if state and state.faces:
                    names = [f.name for f in state.faces if f.name != "unknown person"]
                    unknowns = len([f for f in state.faces if f.name == "unknown person"])
                    
                    parts = []
                    if names: parts.append(", ".join(names))
                    if unknowns: parts.append(f"{unknowns} unrecognized person")
                    
                    if parts:
                        self.response.format_and_speak(f"I see {' and '.join(parts)}.", "HIGH")
                    else:
                        self.response.format_and_speak("There are no people nearby.", "HIGH")
                else:
                    self.response.format_and_speak("I don't see anyone.", "HIGH")
                    
            elif intent == "TEXT_QUERY":
                if state and state.texts:
                    texts = [t.text for t in state.texts]
                    self.response.format_and_speak(f"I can read: {' '.join(texts)}", "HIGH")
                else:
                    self.response.format_and_speak("I don't see any readable text.", "HIGH")
                    
            elif intent == "SAFETY_QUERY":
                if state:
                    is_hazard, _, msg = self.safety.evaluate_safety(state)
                    if is_hazard:
                        self.response.format_and_speak(msg, "CRITICAL")
                    else:
                        self.response.format_and_speak("It looks safe ahead.", "HIGH")
                else:
                    self.response.format_and_speak("I'm not sure, my vision is blocked.", "HIGH")
                        
            elif intent == "FOLLOW_UP_QUERY" or intent == "GENERAL_QUESTION":
                # Use Gemini if API key is set
                if getattr(settings, "GEMINI_API_KEY", "") and state and state.raw_observation:
                    frame = state.raw_observation.get("frame")
                    scene_json = state.raw_observation.get("scene_json")
                    if frame is not None and scene_json:
                        self.response.format_and_speak("Let me think about that...", "NORMAL")
                        import threading
                        is_async_gemini = True
                        def run_gemini():
                            try:
                                ans = self.vision_tools.answer_question(resolved_text, frame, scene_json)
                                self.response.format_and_speak(ans, "HIGH")
                            finally:
                                self.response.mute_background = False
                        threading.Thread(target=run_gemini).start()
                    else:
                        self.response.format_and_speak("I don't have enough visual context right now.", "HIGH")
                else:
                    self.response.format_and_speak("I'm sorry, I can't answer that complex question without my advanced reasoning enabled.", "HIGH")
                    
            self.context_manager.add_turn(text, intent) 
        finally:
            if not is_async_gemini:
                self.response.mute_background = False

_brain = None
def get_assistant_brain():
    global _brain
    if _brain is None:
        _brain = AssistantBrain()
    return _brain
