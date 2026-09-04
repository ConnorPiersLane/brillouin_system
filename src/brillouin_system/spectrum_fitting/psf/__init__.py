"""The camera/detection PSF — measured kernel, modular components.

Structure (single responsibility, no duplicated code):

    components/   one building block per module (Lorentzian, Gaussian,
                  tail, row-tilt boxcar, pixel aperture)
    kernel.py     composes the components into the cached detection
                  kernel; owns the fine grid (DX, PAD_PX)
    profile.py    psf_profile — one Lorentzian through the kernel
    peaks.py      psf1..psf4 — the four per-peak PSFs of the four-peak
                  frame (psf1 outer anti-Stokes + boxcar, psf2/psf3 the
                  inner pair, psf4 outer Stokes + satellite)
    width.py      detected_hwhm_px — the Thompson width chain

The frozen constants (per-peak sigma/tau, the boxcar width, the
satellite ratio/offset) live in the fitting config
(peak_fitting_config.find_peaks_config); the measured record is
peak_fitting_config.psf_measurement.PSF_MEASURED.
"""
from .kernel import DX, PAD_PX, detection_kernel
from .peaks import psf1, psf2, psf3, psf4
from .profile import psf_profile
from .width import detected_hwhm_px

__all__ = ["DX", "PAD_PX", "detection_kernel", "psf_profile",
           "detected_hwhm_px", "psf1", "psf2", "psf3", "psf4"]
