"""One-sided exponential tail — the asymmetric part of the response.

Single responsibility: the sampled one-sided exponential of decay
length tau [px], starting at 0 and pointing toward HIGHER pixel
numbers (the measured direction; a mirrored tail was tested and is
worse than no tail, Data/Figure2/tau_direction_scan.txt). tau <= 0
returns the identity. Same (x0, k) convention as gaussian_kernel.
"""
import numpy as np


def tail_kernel(tau: float, dx: float):
    if tau <= 0:
        return 0.0, np.array([1.0])
    n = max(int(np.ceil(6.0 * tau / dx)), 1)
    k = np.exp(-(dx * np.arange(n + 1)) / tau)
    return 0.0, k
