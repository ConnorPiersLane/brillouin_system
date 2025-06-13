import numpy as np

from brillouin_system.my_dataclasses.calibration import CalibrationCalculator
from brillouin_system.fitting.fit_spectrum_logistic_and_quadratic_bg import get_fitted_spectrum_llq
from brillouin_system.fitting.fit_spectrum_quadratic_bg import get_fitted_spectrum_quadratic_bg
from brillouin_system.fitting.gauss_fitting import get_fitted_spectrum_gaussian
from brillouin_system.fitting.lorentzian_fitting import get_fitted_spectrum_lorentzian
from brillouin_system.fitting.voigt_fitting import get_fitted_spectrum_voigt
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.config.config import find_peaks_reference_config, find_peaks_sample_config




def get_empty_fitting(sline):
    return FittedSpectrum(
        is_success=False,
        x_pixels=np.arange(sline.shape[0]),
        sline=sline,
    )


def fit_reference_spectrum(sline: np.ndarray) -> FittedSpectrum:
    config = find_peaks_reference_config.get()

    if config.fitting_model == 'lorentzian':
        return get_fitted_spectrum_lorentzian(sline=sline, is_reference_mode=True)
    elif config.fitting_model == 'gaussian':
        return get_fitted_spectrum_gaussian(sline=sline, is_reference_mode=True)
    else:
        print(f'[fitting_manager]: unknown fitting_model={config.fitting_model}, fitting failed.')
        return get_empty_fitting(sline)


def fit_sample_spectrum(sline: np.ndarray, calibration_calculator: CalibrationCalculator) -> FittedSpectrum:
    config = find_peaks_sample_config.get()

    if 'voigt' in config.fitting_model and calibration_calculator is None:
        print(f'[fitting_manager]: fitting_model=={config.fitting_model} '
              f'but calibration={calibration_calculator}, no fitting possible.')
        return get_empty_fitting(sline)


    if config.fitting_model == 'lorentzian':
        return get_fitted_spectrum_lorentzian(sline=sline, is_reference_mode=False)
    elif config.fitting_model == 'voigt':
        return get_fitted_spectrum_voigt(sline=sline, is_reference_mode=False,
                                         sigma_func_left=calibration_calculator.sigma_left_peak,
                                         sigma_func_right=calibration_calculator.sigma_right_peak)
    elif config.fitting_model == 'lorentzian_quad_bg':
        return get_fitted_spectrum_quadratic_bg(sline=sline, is_reference_mode=False,
                                                peak_model='lorentzian')
    elif config.fitting_model == 'voigt_quad_bg':
        return get_fitted_spectrum_quadratic_bg(sline=sline, is_reference_mode=False,
                                                peak_model='voigt',
                                                sigma_func_left = calibration_calculator.sigma_left_peak,
                                                sigma_func_right = calibration_calculator.sigma_right_peak)
    elif config.fitting_model == 'lorentzian_log_quad_bg':
        return get_fitted_spectrum_llq(sline=sline, is_reference_mode=False,
                                                peak_model='lorentzian')
    elif config.fitting_model == 'voigt_log_quad_bg':
        return get_fitted_spectrum_llq(sline=sline, is_reference_mode=False,
                                                peak_model='voigt',
                                                sigma_func_left=calibration_calculator.sigma_left_peak,
                                                sigma_func_right=calibration_calculator.sigma_right_peak)
    else:
        print(f'[fitting_manager]: unknown fitting_model={config.fitting_model}, fitting failed.')
        return get_empty_fitting(sline)
