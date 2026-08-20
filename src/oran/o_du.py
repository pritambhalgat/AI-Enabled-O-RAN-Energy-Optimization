class ODU:
    def __init__(self):
        self.scheduler_state = {}

    def schedule(self, allocations):
        self.scheduler_state = allocations
        return allocations
