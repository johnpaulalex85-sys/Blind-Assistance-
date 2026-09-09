import pyttsx3
import threading
import queue
import logging
import time
from config import settings

logger = logging.getLogger(__name__)

PRIORITY_MAP = {
    "CRITICAL": 1,
    "HIGH": 2,
    "NORMAL": 3,
    "LOW": 4,
    "BACKGROUND": 5
}

class TTSManager:
    def __init__(self, rate=None, volume=None):
        self.rate = rate or settings.TTS_RATE
        self.volume = volume or settings.TTS_VOLUME
        self.queue = queue.PriorityQueue()
        self.thread = None
        self.running = False
        self.engine = None
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info("Priority TTS Manager started")
        
    def _worker(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass 
            
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.running = False
            return
            
        while self.running:
            try:
                priority, timestamp, text = self.queue.get(timeout=0.5)
                if text:
                    logger.debug(f"Speaking [{priority}]: {text}")
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS engine error: {e}")
                time.sleep(1)
                
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except ImportError:
            pass

    def speak(self, text: str, priority: str = "NORMAL"):
        if self.running:
            p_val = PRIORITY_MAP.get(priority.upper(), 3)
            # Interrupt if critical
            if p_val == 1:
                self.clear_queue()
                if self.engine:
                    try:
                        self.engine.stop()
                    except:
                        pass
            self.queue.put((p_val, time.time(), text))
        else:
            logger.warning("TTS Manager not running. Cannot speak.")
            
    def clear_queue(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break

    def interrupt(self):
        """Immediately clear the queue and stop the current utterance."""
        if self.running:
            self.clear_queue()
            if self.engine:
                try:
                    self.engine.stop()
                except Exception as e:
                    logger.error(f"Error stopping TTS engine: {e}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("TTS Manager stopped")

tts_manager = TTSManager()
