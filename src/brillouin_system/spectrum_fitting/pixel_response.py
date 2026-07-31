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


def pixel_response_profile(px, amp, cen, gamma, sigma, tau):
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


def make_1pixel_response(sigma, tau):
    """Single-peak model: (amp, cen, gamma, offset)."""
    def model(x, amp, cen, gamma, offset):
        return pixel_response_profile(x, amp, cen, gamma, sigma, tau) + offset
    return model


def make_2pixel_response(sigma, tau_left, tau_right):
    """Two-peak model with the same free parameters as the plain Lorentzian
    pair: (amp1, cen1, wid1, amp2, cen2, wid2, offset).

    Each peak carries its own frozen tail (tau_left for the left/lower-px
    peak, tau_right for the right one) — the two orders were measured to need
    different tail lengths (0.40 vs 0.20 px).
    """
    def model(x, amp1, cen1, wid1, amp2, cen2, wid2, offset):
        return (
            pixel_response_profile(x, amp1, cen1, wid1, sigma, tau_left)
            + pixel_response_profile(x, amp2, cen2, wid2, sigma, tau_right)
            + offset
        )
    return model
