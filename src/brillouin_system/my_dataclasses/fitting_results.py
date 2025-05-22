from dataclasses import dataclass

import numpy as np


@dataclass
class FittingResults:
    frame: np.ndarray
    sline: np.ndarray
    used_rows: list
    inter_peak_distance_px: float
    fitted_spectrum: np.ndarray
    x_pixels: np.ndarray
    x_fit_refined: np.ndarray
    y_fit_refined: np.ndarray
    lorentzian_parameters: np.ndarray
    sd: float
    fsr: float
    freq_shift_ghz: float | None
