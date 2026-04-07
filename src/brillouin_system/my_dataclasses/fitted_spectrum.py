
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





