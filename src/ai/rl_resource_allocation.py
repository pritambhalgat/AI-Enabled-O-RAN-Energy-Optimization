class RLEnvironmentSkeleton:
    def reset(self):
        raise NotImplementedError("Implement state initialization.")

    def step(self, action):
        raise NotImplementedError("Implement transition and reward.")
