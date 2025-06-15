import numpy as np

from brillouin_system.config.config import calibration_config
from brillouin_system.my_dataclasses.analyzer_results import AnalyzerResults
from brillouin_system.my_dataclasses.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum


def compute_freq_shift(fitting: FittedSpectrum, calibration_calculator: CalibrationCalculator) -> float | None:
    if not fitting.is_success or calibration_calculator is None:
        return None

    config = calibration_config.get()

    if config.reference == "left":
        return calibration_calculator.freq_left_peak(fitting.left_peak_center_px)
    elif config.reference == "right":
        return calibration_calculator.freq_right_peak(fitting.right_peak_center_px)
    elif config.reference == "distance":
        return calibration_calculator.freq_peak_distance(fitting.inter_peak_distance)
    else:
        return None

def compute_analyzer_results_from_fitted_spectrum_for_sample(
        frame: np.ndarray,
        fitting: FittedSpectrum,
        calibration_calculator: CalibrationCalculator) -> AnalyzerResults:
    pass
    # if fitting.is_success:
    #
    #     freq_shift_left_peak_ghz = calibration_calculator.
    #     freq_shift_right_peak_ghz: float
    #     freq_shift_peak_distance_ghz: float
    #     psf_sigma_left_peak_ghz: float
    #     psf_sigma_right_peak_ghz: float
    #     fwhm_left_peak_ghz: float
    #     fwhm_right_peak_ghz: float
    #
    #
    #     return DisplayResults(
    #         is_fitting_available=True,
    #         frame=frame,
    #         x_pixels=fitting.x_pixels,
    #         sline=fitting.sline,
    #         x_fit_refined=fitting.x_fit_refined,
    #         y_fit_refined=fitting.y_fit_refined,
    #         inter_peak_distance=fitting.inter_peak_distance,
    #         freq_shift_ghz=freq_shift_ghz,
    #     )
    # else:
    #     return DisplayResults(
    #         is_fitting_available=False,
    #         frame=frame,
    #         x_pixels=fitting.x_pixels,
    #         sline=fitting.sline,
    #     )
    #
    # return AnalyzerResults(
    #     frame = frame,
    #     is_fitting_available = fitting.is_success,
    #     fitted_spectrum: FittedSpectrum,
    #     freq_shift_left_peak_ghz
    #     freq_shift_right_peak_ghz: float
    #     freq_shift_peak_distance_ghz: float
    #     psf_sigma_left_peak_ghz: float
    #     psf_sigma_right_peak_ghz: float
    #     fwhm_left_peak_ghz: float
    #     fwhm_right_peak_ghz: float