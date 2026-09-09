from collections import deque
from config import settings
import re

class ContextManager:
    def __init__(self, max_size: int = settings.CONVERSATION_MEMORY_SIZE):
        self.max_size = max_size
        self.conversation = deque(maxlen=self.max_size)
        self.last_subject = None # e.g. "chair", "Rahul", "the text"

    def add_turn(self, user_text: str, assistant_text: str):
        self.conversation.append({"user": user_text, "assistant": assistant_text})
        
    def update_subject(self, subject: str):
        if subject:
            self.last_subject = subject

    def resolve_pronoun(self, text: str) -> str:
        """
        Naive pronoun resolution. 
        If the user says "what is that" or "who is he", we append the context.
        """
        if not self.last_subject:
            return text
            
        lower_text = text.lower()
        pronouns = ["that", "this", "it", "he", "she", "him", "her", "the person", "the object"]
        
        for pronoun in pronouns:
            pattern = r'\b' + pronoun + r'\b'
            if re.search(pattern, lower_text):
                return text + f" (referring to {self.last_subject})"
        
        return text
