import numpy as np

class SimpleMLPowerController:
    def predict(self, features):
        features = np.asarray(features, dtype=float)
        snr, load = features[0], features[1]
        scale = 0.55 + 0.025 * snr + 0.45 * load
        return float(np.clip(scale, 0.4, 1.0))
