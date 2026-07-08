
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FittedSpectrum:
    """
    This fits the classic two lorentzian peak spectrum from Brilloun
    x: pixels
    y: Brillouin signal
    """
    is_success: bool
    x_pixels: np.ndarray # x axis pixels
    sline: np.ndarray # brillouin signal as function of x (summed over y-pixels)
    model: str = ''
    fitted_spectrum: np.ndarray = field(default=None)
    x_fit_refined: np.ndarray = field(default=None)
    y_fit_refined: np.ndarray = field(default=None)
    mask_for_fitting: np.ndarray = field(default=None)
    parameters: np.ndarray = field(default=None)
    left_peak_center_px: float = None
    left_peak_width_px: float = None
    left_peak_amplitude: float = None
    right_peak_center_px: float = None
    right_peak_width_px: float = None
    right_peak_amplitude: float = None
    inter_peak_distance: float = None
    offset: float = None
    # Anchored DHO fit only (model "2dho_anchored*"): the elastic-line anchor
    # positions used by the fit. For this model the *_peak_center_px fields
    # hold the resonance positions (rayleigh +/- omega), i.e. the true
    # Brillouin shift location, not the visible maximum. None for all other
    # models.
    rayleigh_left_px: float = None
    rayleigh_right_px: float = None
    # Anchored DHO fit only: the MATERIAL HWHM in pixels (instrument response
    # deconvolved), for the loss modulus M''. The *_peak_width_px fields hold
    # the TOTAL observed HWHM (material + instrument), used for photon counts.
    material_hwhm_left_px: float = None
    material_hwhm_right_px: float = None
    # Which calibration chain this fit pairs with when converting px -> GHz:
    # "lorentzian" = the main (Lorentzian-centered) chain, "psf" = the
    # PSF-centered variant. Stamped by SpectrumFitter.fit: plain models are
    # always "lorentzian"; PSF-aware models (lorentzian_psf, dho) inherit the
    # chain of the anchors they were fitted with (ElasticAnchors.chain). A
    # fitted center is only unbiased against the matching chain, so all
    # px->GHz mapping goes through CalibrationCalculator.for_chain. Fits
    # saved before this field existed load as "lorentzian" (correct: they
    # predate the PSF-aware models or used single-chain calibrations).
    calibration_chain: str = "lorentzian"

@dataclass
class GratingSpectrum:
    """
    This fits the spectrum in y-axis, which differs due to the grating.
    y_pixels:
    """
    is_success: bool
    y_pixels: np.ndarray
    sline: np.ndarray # brillouin signal as function of x (summed over y-pixels)
    fitted_spectrum: np.ndarray = field(default=None)
    y_fit_refined: np.ndarray = field(default=None)
    sline_fit_refined: np.ndarray = field(default=None)
    mask_for_fitting: np.ndarray = field(default=None)
    parameters: np.ndarray = field(default=None)
    peak_center_px: float = None
    peak_width_px: float = None
    peak_amplitude: float = None
    offset: float = None





