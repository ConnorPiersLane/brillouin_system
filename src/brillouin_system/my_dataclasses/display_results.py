from dataclasses import dataclass, field

import numpy as np


@dataclass
class DisplayResults:
    is_fitting_available: bool
    frame: np.ndarray
    x_pixels: np.ndarray
    sline: np.ndarray
    mask_for_fitting: np.ndarray = field(default=None)
    x_fit_refined: np.ndarray = field(default=None)
    y_fit_refined: np.ndarray = field(default=None)
    inter_peak_distance: float = None
    freq_shift_ghz: float = None