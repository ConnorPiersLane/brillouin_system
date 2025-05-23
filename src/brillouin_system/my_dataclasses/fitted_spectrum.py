from dataclasses import dataclass

import numpy as np


@dataclass
class FittedSpectrum:
    sline: np.ndarray
    x_pixels: np.ndarray
    fitted_spectrum: np.ndarray
    x_fit_refined: np.ndarray
    y_fit_refined: np.ndarray
    lorentzian_parameters: np.ndarray
    left_peak_pixel: float
    right_peak_pixel: float
    inter_peak_distance: float