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

The measured line is the core through the MEASURED instrument response:

    DHO(nu(x)) (x) kernel

with kernel the profile of the elastic calibration line stacked at this
peak's detector position (spectrum_fitting/epsf.py, per scan or from a
stored fine-sweep table). It already carries the VIPA instrument
Lorentzian, the camera blur, the readout tail and the pixel box, so the
free width IS the acoustic width — no downstream instrument subtraction,
and no camera constants (the parametric kernel was removed 2026-09-10).

Because the fitted center parameter is the RESONANCE pixel, the standard
freq_shift_*_ghz chain downstream reports the damping-corrected resonance
with no new fields (same one-value-chain decision as the 2026-07 anchored
DHO on branch 2dho).

Sample-only: EOM sidebands are elastic laser light with no acoustic mode —
a calibration peak IS the instrument response, so the fitter refuses this
model in reference mode.
"""
from dataclasses import dataclass

import numpy as np

from brillouin_system.spectrum_fitting.measured_kernel import DX, KERNEL_HALF_PX

# Grid padding beyond the evaluated pixels: kernel reach plus margin, so
# edge pixels keep full support.
PAD_PX = KERNEL_HALF_PX + 4.0


@dataclass(frozen=True)
class DhoAxes:
    """What a 'dho_x_psf' sample fit needs from the scan's own calibration.

    freq_*_poly    px -> GHz shift from the peak's own elastic line
                   (np.polyval coefficients), inner pair always, outer
                   orders on four-peak calibrations (None otherwise).
    kernel_*       the measured instrument kernel at each sample peak's
                   position (the first-frame snapshot / capability flag).
    env_slopes     per-scan envelope slopes [1/px] at the sample peaks, in
                   fit order (left, right) or (outer_left, left, right,
                   outer_right); None = the config constants.
    profiles       the scan's Epsf (or epsf.FileKernels): the node table
                   the fitter reads PER FRAME at each peak's found position,
                   so a scan whose shift changes along the way (cornea
                   depth) always uses the profile measured where the peak
                   actually is.

    A FOUR-peak DHO fit (2026-09-05) reports every order's RESONANCE, the
    convention-free quantity, which removes the lineshape-lean systematic
    that makes symmetric-model outer shifts read low by ~Gamma^2/nu_B
    across tracks of different dispersion.
    """
    freq_left_poly: np.ndarray
    freq_right_poly: np.ndarray
    freq_outer_left_poly: np.ndarray | None = None
    freq_outer_right_poly: np.ndarray | None = None
    kernel_left: object | None = None
    kernel_right: object | None = None
    kernel_outer_left: object | None = None
    kernel_outer_right: object | None = None
    env_slopes: tuple | None = None
    profiles: object | None = None

    @property
    def has_measured_kernels(self) -> bool:
        return self.kernel_left is not None and self.kernel_right is not None

    @property
    def has_measured_outer_kernels(self) -> bool:
        return (self.kernel_outer_left is not None
                and self.kernel_outer_right is not None)

    @property
    def has_outer(self) -> bool:
        return (self.freq_outer_left_poly is not None
                and self.freq_outer_right_poly is not None)


def dho_profile(px, amp, cen, gamma_px, freq_poly, kernel):
    """Eq.-S2 DHO through the measured instrument kernel, evaluated at px.

    amp        peak height of the underlying DHO core (before the kernel),
               matching the other models' amplitude convention.
    cen        RESONANCE position [px]: nuB = polyval(freq_poly, cen).
    gamma_px   acoustic HWHM [px]; converted to GHz with the local
               dispersion at cen (Gamma = gamma_px * |d nu/d px|).
    kernel     a MeasuredKernel (unit area on the DX grid) measured at this
               peak's position; it carries the instrument Lorentzian, so
               gamma_px stays the acoustic width.
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

    conv = np.convolve(core, kernel.k) * DX
    conv_x = (xf[0] + kernel.x0) + DX * np.arange(conv.size)

    return float(amp) * np.interp(px, conv_x, conv)
