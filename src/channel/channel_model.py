import numpy as np

def rayleigh_channel(n_users, n_antennas, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return (rng.normal(size=(n_users, n_antennas)) +
            1j * rng.normal(size=(n_users, n_antennas))) / np.sqrt(2)
