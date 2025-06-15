from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum



@dataclass
class AnalyzerResults:
    frame: np.ndarray
    is_fitting_available: bool
    fitted_spectrum: FittedSpectrum
    freq_shift_left_peak_ghz: float
    freq_shift_right_peak_ghz: float
    freq_shift_peak_distance_ghz: float
    psf_sigma_left_peak_ghz: float
    psf_sigma_right_peak_ghz: float
    fwhm_left_peak_ghz: float
    fwhm_right_peak_ghz: float