import numpy as np

def greedy_energy_optimizer(sinr_targets, max_power):
    targets = np.asarray(sinr_targets, dtype=float)
    weights = np.maximum(targets, 1e-9)
    return max_power * weights / weights.sum()
