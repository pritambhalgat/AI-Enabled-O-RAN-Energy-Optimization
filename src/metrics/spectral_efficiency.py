import numpy as np

def sum_spectral_efficiency(sinr_linear):
    return float(np.sum(np.log2(1.0 + np.asarray(sinr_linear))))
