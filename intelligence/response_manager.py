class ResponseManager:
    def __init__(self, audio_tools):
        self.audio = audio_tools
        self.last_spoken = {}
        import time
        self.time = time
        self.pending_messages = []
        self.last_speech_time = 0.0
        self.mute_background = False

    def format_and_speak(self, message: str, priority: str = "NORMAL"):
        """
        Formats response and sends to Priority TTS, filtering out rapid repetitions.
        """
        now = self.time.time()
        
        # 1. Handle global cooldown and mute for background chatter
        if priority not in ["CRITICAL", "HIGH"]:
            if self.mute_background:
                return
            if now - self.last_speech_time < 5.0:
                return

        # 2. Deduplication based on exact message
        if message in self.last_spoken:
            time_since = now - self.last_spoken[message]
            if priority in ["CRITICAL", "HIGH"]:
                if time_since < 5.0: # 5 second cooldown for exact identical critical messages
                    return
            else:
                if time_since < 15.0: # 15 second cooldown for exact identical normal messages
                    return
        
        self.last_spoken[message] = now
        self.last_speech_time = now
        self.pending_messages.append(message)
        self.audio.speak(message, priority)

    def get_pending_messages(self):
        msgs = self.pending_messages[:]
        self.pending_messages.clear()
        return msgs
