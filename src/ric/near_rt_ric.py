class NearRTRIC:
    def __init__(self, xapp):
        self.xapp = xapp

    def control(self, telemetry):
        return self.xapp.decide(telemetry)
