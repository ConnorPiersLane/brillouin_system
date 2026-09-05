"""The camera/detection PSF — measured kernel, modular components.

PRODUCTION model (one family, identical for all four peaks):

    Lorentzian(gamma) ⊗ Gauss(sigma) ⊗ one-sided ExpTail(tau) ⊗ Pixel

with one frozen (sigma, tau) per peak from the fitting config.

Structure (single responsibility, no duplicated code):

    components/   one building block per module (Lorentzian, Gaussian,
                  tail, pixel aperture)
    kernel.py     composes the components into the cached detection
                  kernel; owns the fine grid (DX, PAD_PX)
    profile.py    psf_profile — one Lorentzian through the kernel; THE
                  per-peak model of every production fit
    width.py      detected_hwhm_px — the Thompson width chain
    extras.py     measured-but-NOT-production terms (row-tilt boxcar,
                  outer_right satellite, psf1/psf4), deliberately
                  separated from the chain (user decision 2026-09-04)

The frozen constants live in the fitting config
(peak_fitting_config.find_peaks_config); the measured record —
including the extras' constants — is
peak_fitting_config.psf_measurement.PSF_MEASURED.
"""
from .kernel import DX, PAD_PX, detection_kernel
from .profile import psf_profile
from .width import detected_hwhm_px

__all__ = ["DX", "PAD_PX", "detection_kernel", "psf_profile",
           "detected_hwhm_px"]
