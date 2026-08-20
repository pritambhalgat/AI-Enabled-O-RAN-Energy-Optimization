import numpy as np

def shannon_sum_rate(sinr_linear, bandwidth_hz):
    return float(bandwidth_hz * np.sum(np.log2(1.0 + np.asarray(sinr_linear))))
