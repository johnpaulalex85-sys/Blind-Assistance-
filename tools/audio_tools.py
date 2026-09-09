from audio.tts import tts_manager

class AudioTools:
    def speak(self, text: str, priority: str = "NORMAL"):
        tts_manager.speak(text, priority)
        
    def stop(self):
        tts_manager.clear_queue()
        
    def interrupt(self):
        tts_manager.interrupt()
        
    def clear_queue(self):
        tts_manager.clear_queue()
