import numpy as np

def equal_power(n_users, total_power):
    return np.full(n_users, total_power / n_users)

def energy_aware_power(channel_gains, total_power, minimum_fraction=0.05):
    gains = np.maximum(np.asarray(channel_gains), 1e-12)
    weights = gains / gains.sum()
    weights = minimum_fraction / len(gains) + (1 - minimum_fraction) * weights
    return total_power * weights / weights.sum()
