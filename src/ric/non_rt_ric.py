class NonRTRIC:
    def __init__(self, rapp):
        self.rapp = rapp

    def policy(self, history):
        return self.rapp.generate_policy(history)
