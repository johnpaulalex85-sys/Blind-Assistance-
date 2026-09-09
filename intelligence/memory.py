import time
from collections import deque
from typing import List
from scene.scene_state import SceneState
from config import settings

class SceneMemory:
    def __init__(self, duration_sec: int = settings.SCENE_MEMORY_DURATION_SEC):
        self.duration_sec = duration_sec
        # Store tuples of (timestamp, SceneState)
        self.history: deque = deque()

    def add_state(self, state: SceneState):
        current_time = time.time()
        self.history.append((current_time, state))
        self._prune(current_time)

    def _prune(self, current_time: float):
        while self.history and current_time - self.history[0][0] > self.duration_sec:
            self.history.popleft()

    def get_recent_objects(self) -> List[str]:
        # Return a unique list of class_names seen recently
        objects = set()
        for _, state in self.history:
            for obj in state.objects:
                objects.add(obj.class_name)
        return list(objects)

    def get_recent_people(self) -> List[str]:
        people = set()
        for _, state in self.history:
            for face in state.faces:
                if face.name != "unknown person":
                    people.add(face.name)
        return list(people)
        
    def get_latest_state(self) -> SceneState:
        if self.history:
            return self.history[-1][1]
        return None
