"""Detected width of a fitted peak — the Thompson-bound width chain.

Single responsibility: convert a fitted Lorentzian core gamma to the
HWHM of the profile as it lands on the detector,
Lorentzian(gamma) ⊗ Gauss(sigma) ⊗ Tail(tau).

The pixel top-hat is deliberately EXCLUDED: in the Thompson bound the
binning is already the separate a^2/12/N pixelation term, so folding
it into s as well would double-count it. With sigma = tau = 0 this
reduces to gamma exactly.

Cost: the convolution runs on the fine grid (~95 us), but gamma is
cached at 0.001 px resolution (<< the bound's own precision) and
sigma/tau are frozen constants, so within a scan almost every call is
a cache hit.
"""
from functools import lru_cache

import numpy as np

from .components import gaussian_kernel, tail_kernel
from .kernel import DX


def detected_hwhm_px(gamma, sigma, tau) -> float:
    """HWHM [px] of the peak as it lands on the detector.

    The fitted gamma is the Lorentzian CORE before the camera PSF; the
    photons, however, arrive spread by the detection kernel — this
    returns that profile's HWHM (half of its full width at half
    maximum; the tail makes it asymmetric).
    """
    gamma = max(float(gamma), 1e-9)
    sigma = float(sigma)
    tau = float(tau)
    if sigma <= 0.0 and tau <= 0.0:
        return gamma
    return _detected_hwhm_cached(int(round(gamma * 1000.0)), sigma, tau)


@lru_cache(maxsize=4096)
def _detected_hwhm_cached(gamma_millipx: int, sigma: float,
                          tau: float) -> float:
    gamma = max(gamma_millipx / 1000.0, 1e-9)

    half = 10.0 * gamma + 6.0 * (sigma + tau) + 1.0
    n = int(round(2.0 * half / DX)) + 1
    x = -half + DX * np.arange(n)
    prof = 1.0 / (1.0 + (x / gamma) ** 2)
    x0 = x[0]

    for k_x0, k in (gaussian_kernel(sigma, DX), tail_kernel(tau, DX)):
        if k.size > 1:
            prof = np.convolve(prof, k / k.sum())
            x0 += k_x0

    xs = x0 + DX * np.arange(prof.size)
    i_max = int(np.argmax(prof))
    half_max = prof[i_max] / 2.0
    left = np.interp(half_max, prof[: i_max + 1], xs[: i_max + 1])
    right = np.interp(half_max, prof[i_max:][::-1], xs[i_max:][::-1])
    return float((right - left) / 2.0)
