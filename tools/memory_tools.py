class MemoryTools:
    def __init__(self, scene_memory, context_manager):
        self.scene_memory = scene_memory
        self.context = context_manager

    def get_recent_objects(self):
        return self.scene_memory.get_recent_objects()

    def get_recent_people(self):
        return self.scene_memory.get_recent_people()
        
    def search_context(self, text: str) -> str:
        return self.context.resolve_pronoun(text)
