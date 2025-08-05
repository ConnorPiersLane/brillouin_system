
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FittedSpectrum:
    is_success: bool
    x_pixels: np.ndarray
    sline: np.ndarray
    model: str = ''
    fitted_spectrum: np.ndarray = field(default=None)
    x_fit_refined: np.ndarray = field(default=None)
    y_fit_refined: np.ndarray = field(default=None)
    parameters: np.ndarray = field(default=None)
    left_peak_center_px: float = None
    left_peak_width_px: float = None
    left_peak_amplitude: float = None
    right_peak_center_px: float = None
    right_peak_width_px: float = None
    right_peak_amplitude: float = None
    inter_peak_distance: float = None





