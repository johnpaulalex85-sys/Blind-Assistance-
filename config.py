import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FRIDAY"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Model Paths / Configs
    YOLO_MODEL_PATH: str = "models/yolov8n.pt"
    INSIGHTFACE_MODEL_DIR: str = "models/insightface"
    DEPTH_MODEL_NAME: str = "depth-anything/Depth-Anything-V2-Small-hf"
    QWEN_MODEL_NAME: str = "Qwen/Qwen2-VL-2B-Instruct"
    WHISPER_MODEL_SIZE: str = "base"
    GEMINI_API_KEY: str = ""
    
    # Camera Configs
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    
    # Audio Configs
    TTS_RATE: int = 150
    TTS_VOLUME: float = 1.0
    
    # Perception Configs
    YOLO_CONF_THRESHOLD: float = 0.4
    OCR_CONF_THRESHOLD: float = 0.5
    FACE_CONF_THRESHOLD: float = 0.6
    
    # Tracker Configs
    TRACKER_MAX_AGE: int = 30
    TRACKER_MIN_HITS: int = 3
    TRACKER_IOU_THRESHOLD: float = 0.3
    
    # Memory Configs
    SCENE_MEMORY_DURATION_SEC: int = 60
    CONVERSATION_MEMORY_SIZE: int = 10
    
    # Safety Configs
    SAFETY_CRITICAL_DISTANCE: float = 1.5 # pseudo-meters
    
    # Assistant Modes
    ASSISTANT_MODE: str = "NORMAL" # NORMAL, QUIET, SAFETY, FOCUS
    ENABLE_QWEN: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
