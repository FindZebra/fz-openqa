import psutil


class MemoryRequirement:
    def __init__(self, memory: float):
        self.memory = memory

    def __call__(self) -> bool:
        mem = psutil.virtual_memory()
        available_mem = mem.available / 1e9
        return available_mem > self.memory

    def explain(self):
        mem = psutil.virtual_memory()
        available_mem = mem.available / 1e9
        req = self.__call__()
        if req:
            return (
                f"Memory requirements: {self.memory} GB are met "
                f"(available = {available_mem:.1f} GB)"
            )
        else:
            return (
                f"Memory requirements: {self.memory} GB not met "
                f"(available = {available_mem:.1f} GB)"
            )
