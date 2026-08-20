class EnergyOptimizationXApp:
    # Lightweight deterministic baseline xApp.
    def decide(self, telemetry):
        load = telemetry.get("traffic_load", 1.0)
        snr = telemetry.get("snr_db", 10.0)
        if load < 0.4 and snr > 10:
            return {"power_scale": 0.65}
        if load < 0.7:
            return {"power_scale": 0.85}
        return {"power_scale": 1.0}
