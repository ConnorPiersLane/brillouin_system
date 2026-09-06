"""MEASURED-BUT-NOT-PRODUCTION detection terms — kept OUT of the chain.

USER DECISION (2026-09-04, end of the boxcar arc): production fits use
ONE kernel family for all four peaks — Lorentzian(gamma) x Gauss(sigma)
x one-sided ExpTail(tau) x Pixel — with one frozen (sigma, tau) per
peak and nothing else. The two extra terms measured during the 09-02..
09-04 campaigns are preserved HERE, clearly separated, for analyses
that want them; the production fitter, config and TOML know nothing of
them.

    ROW-TILT BOXCAR  boxcar_kernel / detection_kernel_with_box / psf1
        Every line is tilted against the CCD columns by ONE physical
        constant, a ~27 MHz/row frequency shear (a VIPA property, not
        camera rotation) divided by each track's local dispersion —
        measured stable to 0.1-3% across all four 9-2 fine sweeps incl.
        a realignment (Data/2026-9-2/tilt_all_runs.py). Over the 13-row
        band that is a top-hat of tilt x 13 px: 1.95 / 1.25 / 1.02 /
        0.85 for [outer_left, left, right, outer_right]. Only the OUTER
        orders ever preferred the literal top-hat (outer_left's
        trapezoid profile; outer_right sine 0.54 -> 0.13 MHz with
        tau 0.08 +- 0.001 across four runs); the INNER peaks' bell-
        weighted smear is Gauss+tail-shaped and lives in their
        sigma/tau (a top-hat there ruins their sines 0.09/0.25 ->
        4.8/2.9 MHz). Full record: psf_measurement.PSF_MEASURED and the
        determine_allbox summaries with the 9-2 data.

    OUTER-RIGHT SATELLITE  psf4
        An intrinsic near-core satellite of that VIPA order — a scaled
        displaced copy of the line (ratio 0.037 at -1.23 px), blind-
        validated across sweeps (Data/2026-9-2/ghost_model_opt*).
        WITHOUT it the outer_right fitted position wobbles once per
        pixel by ~3.2-3.6 MHz (re-measured 09-04 with the box: same);
        production accepts that wobble in exchange for the plain
        kernel — outer_right positions are precision-grade only with
        this term.

    psf1 outer anti-Stokes with its boxcar; psf4 outer Stokes with
    boxcar + satellite. The inner peaks never had extra terms (their
    former psf2/psf3 were plain wrappers and live on as psf_profile).

Constants are passed explicitly — nothing here reads the fitting
config.
"""
import numpy as np

from .components import (boxcar_kernel, gaussian_kernel, lorentzian,
                         pixel_kernel, tail_kernel)
from .kernel import DX, PAD_PX

# NOTE 2026-09-05: the outer-order boxcar and the outer_right satellite
# were RESTORED to production (config fields psf_box_outer_*_px and
# psf_sat_*; the kernel chain takes box directly). This module remains
# for analyses that need the terms standalone — e.g. trying a box on
# the INNER peaks, which production deliberately refuses.


def detection_kernel_with_box(sigma: float, tau: float, dx: float,
                              box: float):
    """(x0, k): the production kernel with the row-tilt boxcar folded in."""
    g_x0, g = gaussian_kernel(sigma, dx)
    t_x0, t = tail_kernel(tau, dx)
    b_x0, b = pixel_kernel(dx)
    w_x0, w = boxcar_kernel(box, dx)
    k = np.convolve(np.convolve(np.convolve(g, t), b), w)
    k = k / (k.sum() * dx)
    return g_x0 + t_x0 + b_x0 + w_x0, k


def psf_profile_with_box(px, amp, cen, gamma, sigma, tau, box):
    """Lorentzian(gamma) through the boxcar-extended kernel."""
    px = np.asarray(px, dtype=float)
    lo = float(px.min()) - PAD_PX
    hi = float(px.max()) + PAD_PX
    n = int(round((hi - lo) / DX)) + 1
    xf = lo + DX * np.arange(n)
    lor = lorentzian(xf, cen, gamma)
    k_x0, k = detection_kernel_with_box(float(sigma), float(tau), DX,
                                        float(box))
    conv = np.convolve(lor, k) * DX
    conv_x = (xf[0] + k_x0) + DX * np.arange(conv.size)
    return float(amp) * np.interp(px, conv_x, conv)


def psf1(px, amp, cen, gamma, sigma, tau, box):
    """Outer anti-Stokes order: kernel + measured row-tilt boxcar."""
    return psf_profile_with_box(px, amp, cen, gamma, sigma, tau, box)


def psf4(px, amp, cen, gamma, sigma, tau, sat_ratio, sat_delta,
         box=0.0):
    """Outer Stokes order: kernel [+ boxcar] + intrinsic satellite."""
    main = psf_profile_with_box(px, amp, cen, gamma, sigma, tau, box)
    if sat_ratio > 0.0:
        main = main + psf_profile_with_box(px, amp * sat_ratio,
                                           cen + sat_delta, gamma,
                                           sigma, tau, box)
    return main
