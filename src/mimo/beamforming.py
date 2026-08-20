import numpy as np

def mrt_precoder(H):
    # Maximum-ratio transmission precoder.
    W = H.conj().T
    norms = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    return W / norms
