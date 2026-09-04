"""The four per-peak detection PSFs, left to right on the frame.

The four-peak ROI shows two VIPA order pairs; every peak shares ONE
model family — Lorentzian ⊗ Gauss ⊗ Tail [⊗ Boxcar] ⊗ Pixel — and each
carries its own frozen constants from the fitting config:

    psf1  OUTER anti-Stokes order (outermost LEFT peak). Its line
          tilt is the largest on the frame (0.147 px/row), so the
          13-row sum smears it with a ~1.9 px top-hat — measured
          geometry, and directly visible as the flat top of its
          stacked profile.
    psf2  INNER anti-Stokes (main LEFT peak of the pair the two-peak
          fit uses).
    psf3  INNER Stokes (main RIGHT peak).
    psf4  OUTER Stokes order (outermost RIGHT peak). Carries the
          intrinsic near-core SATELLITE: a scaled displaced copy of
          the line (ratio ~3.7% at -1.23 px, an optical property of
          that order, blind-validated across sweeps) — without it the
          fitted position wobbles once per pixel by ~3 MHz.

Every peak takes a row-tilt boxcar width (its own measured tilt x the
row-band height; 0 = off): the tilt smear exists for all four, the
peaks differ only in whether it is big enough to matter.

Fitted per frame: amp, cen, gamma (plus the per-peak flat offset in
the fit model). Everything else is a frozen instrument constant.
"""
from .profile import psf_profile


def psf1(px, amp, cen, gamma, sigma, tau, box):
    """Outer anti-Stokes order: kernel + measured row-tilt boxcar."""
    return psf_profile(px, amp, cen, gamma, sigma, tau, box=box)


def psf2(px, amp, cen, gamma, sigma, tau, box=0.0):
    """Inner anti-Stokes (main left peak)."""
    return psf_profile(px, amp, cen, gamma, sigma, tau, box=box)


def psf3(px, amp, cen, gamma, sigma, tau, box=0.0):
    """Inner Stokes (main right peak)."""
    return psf_profile(px, amp, cen, gamma, sigma, tau, box=box)


def psf4(px, amp, cen, gamma, sigma, tau, sat_ratio, sat_delta, box=0.0):
    """Outer Stokes order: kernel + intrinsic near-core satellite."""
    main = psf_profile(px, amp, cen, gamma, sigma, tau, box=box)
    if sat_ratio > 0.0:
        main = main + psf_profile(px, amp * sat_ratio, cen + sat_delta,
                                  gamma, sigma, tau, box=box)
    return main
