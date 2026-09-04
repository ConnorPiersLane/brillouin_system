"""Damped-harmonic-oscillator sample lineshape (model 'dho_x_psf').

The core is eq. S2 of Bailey et al., Sci. Adv. 6, eabc1937 (2020),
supplementary materials:

    I(nu) = I0 * nuB^2 * Gf / ((nuB^2 - nu^2)^2 + (nu*Gf)^2)

written in nu = FREQUENCY SHIFT from the peak's OWN elastic (Rayleigh) line,
with nuB the acoustic resonance and Gf = 2*Gamma the full damping width
(Gamma = HWHM of the equivalent Lorentzian near resonance). The function is
even in nu — mirror-symmetric about the elastic line — and its maximum sits
at nu^2 = nuB^2 - Gf^2/2, BELOW the resonance: the heavy wing points toward
lower shift. On the detector the two peaks' shift axes run in OPPOSITE pixel
directions (the visible pair is Stokes of order n + anti-Stokes of order
n+1, elastic lines on opposite sides), so the lean lands on opposite pixel
sides per peak. That is not hand-coded anywhere: the core is built in
nu(x) = polyval(freq_poly, x) using each peak's own calibration track, and
the polys' opposite slopes place it (design validated 2026-08-05/10,
Data/2026-8-5/analysis/dho_vs_lorentzian.py — synthetic closure returns an
injected resonance to 0.00 MHz on both peaks).

The measured line is the core through the full instrument chain:

    DHO(nu(x)) (x) Lorentzian(g_inst) (x) Gauss(sigma) (x) ExpTail(tau) (x) pixel

g_inst is the VIPA instrument Lorentzian HWHM [px] from the calibration
width polynomial, evaluated at the peak's position and FIXED in the fit.
Folding it into the kernel is essential, not cosmetic: the DHO's center
offset scales with the square of the width that drives the asymmetry, and
leaving the instrument width inside the free parameter makes that offset
~4x too large (measured 2026-08-05). With it in the kernel the free width
IS the acoustic width — no downstream instrument subtraction.

Because the fitted center parameter is the RESONANCE pixel, the standard
freq_shift_*_ghz chain downstream reports the damping-corrected resonance
with no new fields (same one-value-chain decision as the 2026-07 anchored
DHO on branch 2dho).

Sample-only: EOM sidebands are elastic laser light with no acoustic mode —
a calibration peak IS the instrument response, so the fitter refuses this
model in reference mode.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from brillouin_system.spectrum_fitting.psf import DX, detection_kernel

# Half-width [px] of the instrument-Lorentzian kernel. Lorentzian tails are
# long; 12 px truncates ~2% of the area for the typical g_inst ~0.4 px, flat
# enough to be absorbed by the fitted offset. Same value as the validated
# 2026-08 analysis harness.
KERNEL_HALF_PX = 12.0
# Grid padding beyond the evaluated pixels: kernel reach (Lorentzian half
# width + Gauss/tail extent) plus margin, so edge pixels keep full support.
PAD_PX = KERNEL_HALF_PX + 4.0


@dataclass(frozen=True)
class DhoAxes:
    """Per-peak calibration inputs a 'dho_x_psf' fit needs.

    Built from the scan's own calibration (CalibrationCalculator.dho_axes())
    and passed to SpectrumFitter.fit(dho_axes=...) — the same pattern as the
    reflection background. Inner main pair only: the calibration stores no
    width tracks for the outer orders (their readout taus are provisional —
    positions yes, width claims no, 2026-08-20), and the DHO center
    correction scales as Gamma^2, so a biased width input would land
    directly in the resonance.
    """
    # px -> GHz shift from the peak's own elastic line (np.polyval coeffs).
    freq_left_poly: np.ndarray
    freq_right_poly: np.ndarray
    # px -> instrument Lorentzian HWHM [px] (the calibration width polys).
    instrument_width_left_poly: np.ndarray
    instrument_width_right_poly: np.ndarray


@lru_cache(maxsize=64)
def _dho_kernel(g_inst_millipx: int, sigma: float, tau: float):
    """Lorentzian(g_inst) (x) Gauss(sigma) (x) ExpTail(tau) (x) pixel.

    Returns (x0, k) with k normalised to unit area (k.sum()*DX == 1) and x0
    the coordinate of k[0] relative to the kernel centre. g_inst is keyed in
    milli-px: it is frozen per peak per scan (evaluated at the found peak
    position before the fit), so within a scan every call is a cache hit.
    """
    g = max(g_inst_millipx / 1000.0, 1e-6)
    n = int(round(KERNEL_HALF_PX / DX))
    xk = DX * (np.arange(2 * n + 1) - n)
    lor = 1.0 / (1.0 + (xk / g) ** 2)
    lor /= lor.sum()

    cam_x0, cam = detection_kernel(float(sigma), float(tau), DX)  # unit area
    k = np.convolve(lor, cam)
    return -KERNEL_HALF_PX + cam_x0, k


def dho_profile(px, amp, cen, gamma_px, freq_poly, g_inst_px, sigma, tau):
    """Eq.-S2 DHO through the instrument chain, evaluated at pixels px.

    amp       peak height of the underlying DHO core (before the kernel),
              matching the other models' amplitude convention.
    cen       RESONANCE position [px]: nuB = polyval(freq_poly, cen).
    gamma_px  acoustic HWHM [px]; converted to GHz with the local dispersion
              at cen (Gamma = gamma_px * |d nu/d px|).
    """
    px = np.asarray(px, dtype=float)
    gamma_px = max(float(gamma_px), 1e-9)

    lo = float(px.min()) - PAD_PX
    hi = float(px.max()) + PAD_PX
    n = int(round((hi - lo) / DX)) + 1
    xf = lo + DX * np.arange(n)

    nu = np.polyval(freq_poly, xf)
    nu_b = float(np.polyval(freq_poly, cen))
    slope = float(np.polyval(np.polyder(freq_poly), cen))
    gam = gamma_px * abs(slope)          # acoustic HWHM [GHz]
    gf = 2.0 * gam                       # DHO full damping width

    core = gf * nu_b ** 2 / ((nu ** 2 - nu_b ** 2) ** 2 + (gf * nu) ** 2)
    # Peak value in closed form (maximum at nu^2 = nuB^2 - gf^2/2), so amp is
    # the core height; grid-max fallback only for the overdamped corner the
    # bounds should never reach.
    denom = gf * (nu_b ** 2 - 0.25 * gf ** 2)
    if nu_b ** 2 > 0.5 * gf ** 2 and denom > 0.0:
        core_max = nu_b ** 2 / denom
    else:
        core_max = float(np.max(core))
    core = core / max(core_max, 1e-300)

    k_x0, k = _dho_kernel(int(round(float(g_inst_px) * 1000.0)),
                          float(sigma), float(tau))
    conv = np.convolve(core, k) * DX
    conv_x = (xf[0] + k_x0) + DX * np.arange(conv.size)

    return float(amp) * np.interp(px, conv_x, conv)
