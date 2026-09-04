"""Gaussian blur — the symmetric optical / charge-diffusion PSF core.

Single responsibility: the sampled Gaussian kernel of r.m.s. width
sigma [px]. Kernel convention shared by all components: returns
(x0, k) where x0 is the coordinate of k[0] relative to the profile
centre and k is UNNORMALISED (the composer normalises the full chain
once); sigma <= 0 returns the identity.
"""
import numpy as np


def gaussian_kernel(sigma: float, dx: float):
    if sigma <= 0:
        return 0.0, np.array([1.0])
    half = max(int(np.ceil(4.0 * sigma / dx)), 1)
    k = np.exp(-0.5 * ((dx * np.arange(-half, half + 1)) / sigma) ** 2)
    return -half * dx, k
