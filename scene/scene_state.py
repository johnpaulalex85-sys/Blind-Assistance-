from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    box: List[float] # x1, y1, x2, y2
    distance_category: str = "UNKNOWN"
    distance_val: float = 0.0

@dataclass
class DetectedFace:
    name: str
    box: List[float]
    confidence: float

@dataclass
class DetectedText:
    text: str
    box: List[float]

@dataclass
class SceneState:
    timestamp: float
    objects: List[TrackedObject] = field(default_factory=list)
    faces: List[DetectedFace] = field(default_factory=list)
    texts: List[DetectedText] = field(default_factory=list)
    raw_observation: Dict[str, Any] = field(default_factory=dict)
