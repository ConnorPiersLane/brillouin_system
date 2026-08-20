"""
Camera pixel-response lineshape:

    measured line = Lorentzian(gamma) (x) Gauss(sigma) (x) ExpTail(tau) (x) pixel

sigma  Gaussian charge diffusion: photo-electrons wander sideways in the
       silicon before a pixel collects them. Symmetric, ~0.25 px.
tau    one-sided exponential readout smear: each charge transfer leaves a
       small fraction behind, which arrives one or more shifts late — always
       toward the readout direction (higher pixel numbers here). This is the
       asymmetric part; a Voigt profile cannot represent it.

sigma and tau are FROZEN instrument constants supplied by the config, not
free parameters: the fit keeps the same amplitude/centre/width per peak as
the plain Lorentzian model, so initial guesses and bounds transfer directly.
Measured 2026-07 on the fine EOM sweeps as sigma 0.25, tau_left 0.40,
tau_right 0.20 px; that removed the +-7 MHz one-cycle-per-pixel residual
sine on six calibrations spanning seven weeks.

The kernel depends only on (sigma, tau) and the grid step, so it is built
once and cached; each evaluation is one Lorentzian on a fine grid plus one
convolution.
"""
from functools import lru_cache

import numpy as np

# Fine grid step [px] for the convolution. The frozen constants were measured
# with 0.02 px; kernel features are ~0.25 px wide, so this is well sampled.
DX = 0.02
# Grid padding beyond the evaluated pixels so the tail is not truncated.
PAD_PX = 4.0


@lru_cache(maxsize=32)
def _kernel(sigma: float, tau: float, dx: float):
    """Gauss(sigma) (x) ExpTail(tau) (x) pixel on the fine grid.

    Returns (x0, k) where k is normalised to unit AREA (k.sum()*dx == 1) and
    x0 is the coordinate of k[0] relative to the profile centre. Coordinates
    are tracked explicitly: convolving arrays sampled from a0 and b0 yields a
    grid starting at a0+b0, so the pixel top-hat (which spans -0.5..+0.5) and
    the one-sided tail (which starts at 0) end up correctly placed instead of
    silently shifting the profile.
    """
    # Gaussian charge diffusion, centred on 0
    if sigma > 0:
        half = max(int(np.ceil(4.0 * sigma / dx)), 1)
        g = np.exp(-0.5 * ((dx * np.arange(-half, half + 1)) / sigma) ** 2)
        g_x0 = -half * dx
    else:
        g = np.array([1.0])
        g_x0 = 0.0

    # One-sided exponential readout tail, starting at 0 toward higher px
    if tau > 0:
        n_t = max(int(np.ceil(6.0 * tau / dx)), 1)
        t = np.exp(-(dx * np.arange(n_t + 1)) / tau)
        t_x0 = 0.0
    else:
        t = np.array([1.0])
        t_x0 = 0.0

    # Pixel top-hat spanning -0.5..+0.5
    n_b = max(int(round(1.0 / dx)), 1)
    b = np.ones(n_b + 1)
    b_x0 = -0.5

    k = np.convolve(np.convolve(g, t), b)
    k = k / (k.sum() * dx)
    return g_x0 + t_x0 + b_x0, k


def psf_profile(px, amp, cen, gamma, sigma, tau):
    """Lorentzian(gamma) through the pixel response, evaluated at px.

    amp is the peak height of the underlying Lorentzian, matching the plain
    pixel-integrated Lorentzian model's convention.
    """
    px = np.asarray(px, dtype=float)
    gamma = max(float(gamma), 1e-12)

    lo = float(px.min()) - PAD_PX
    hi = float(px.max()) + PAD_PX
    n = int(round((hi - lo) / DX)) + 1
    xf = lo + DX * np.arange(n)

    lor = 1.0 / (1.0 + ((xf - float(cen)) / gamma) ** 2)

    k_x0, k = _kernel(float(sigma), float(tau), DX)
    conv = np.convolve(lor, k) * DX
    conv_x0 = xf[0] + k_x0
    conv_x = conv_x0 + DX * np.arange(conv.size)

    return float(amp) * np.interp(px, conv_x, conv)


def detected_hwhm_px(gamma, sigma, tau) -> float:
    """HWHM [px] of the peak as it lands on the detector.

    The fitted gamma is the Lorentzian CORE before the camera PSF; the
    photons, however, arrive spread by Lorentzian(gamma) (x) Gauss(sigma)
    (x) ExpTail(tau) — this returns that profile's HWHM (half of its full
    width at half maximum; the tail makes it asymmetric).

    The pixel top-hat is deliberately EXCLUDED: in the Thompson bound the
    binning is already the separate a^2/12/N pixelation term, so folding it
    into s as well would double-count it. With sigma = tau = 0 this reduces
    to gamma exactly.

    Cost: the convolution runs on the fine grid (~95 us), but gamma is
    cached at 0.001 px resolution (<< the bound's own precision) and
    sigma/tau are frozen constants, so within a scan almost every call is a
    cache hit. Uncached it is still ~0.4% of the curve_fit it accompanies.
    """
    gamma = max(float(gamma), 1e-9)
    sigma = float(sigma)
    tau = float(tau)
    if sigma <= 0.0 and tau <= 0.0:
        return gamma
    return _detected_hwhm_cached(int(round(gamma * 1000.0)), sigma, tau)


@lru_cache(maxsize=4096)
def _detected_hwhm_cached(gamma_millipx: int, sigma: float, tau: float) -> float:
    gamma = max(gamma_millipx / 1000.0, 1e-9)

    half = 10.0 * gamma + 6.0 * (sigma + tau) + 1.0
    n = int(round(2.0 * half / DX)) + 1
    x = -half + DX * np.arange(n)
    prof = 1.0 / (1.0 + (x / gamma) ** 2)
    x0 = x[0]

    if sigma > 0.0:
        m = max(int(np.ceil(4.0 * sigma / DX)), 1)
        g = np.exp(-0.5 * ((DX * np.arange(-m, m + 1)) / sigma) ** 2)
        prof = np.convolve(prof, g / g.sum())
        x0 += -m * DX
    if tau > 0.0:
        m = max(int(np.ceil(6.0 * tau / DX)), 1)
        t = np.exp(-(DX * np.arange(m + 1)) / tau)
        prof = np.convolve(prof, t / t.sum())

    xs = x0 + DX * np.arange(prof.size)
    i_max = int(np.argmax(prof))
    half_max = prof[i_max] / 2.0
    left = np.interp(half_max, prof[: i_max + 1], xs[: i_max + 1])
    right = np.interp(half_max, prof[i_max:][::-1], xs[i_max:][::-1])
    return float((right - left) / 2.0)
