import numpy as np
from scipy.optimize import curve_fit

from brillouin_system.fitting.lorentzian_fitting import _2Lorentzian
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.fitting.gauss_fitting import _2Gaussian
from brillouin_system.fitting.voigt_fitting import _2Voigt
from brillouin_system.fitting.fit_util import refine_fitted_spectrum, sort_peaks, find_peak_locations, \
    select_top_two_peaks


def _quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

def _fit_background_quadratic(px, sline):
    p0 = [0.0, 0.0, 0.0]
    popt, _ = curve_fit(
        _quadratic,
        px,
        sline,
        p0=p0,
        maxfev=10000
    )
    return popt

def _spectrum_function_quadratic_bg(x, peak_model, sigma_func_left, sigma_func_right, *para):
    p_peaks = para[:7]
    p_bg = para[7:]

    if peak_model == 'lorentzian':
        return _2Lorentzian(x, *p_peaks) + _quadratic(x, *p_bg)
    elif peak_model == 'gaussian':
        return _2Gaussian(x, *p_peaks) + _quadratic(x, *p_bg)
    elif peak_model == 'voigt':
        return _2Voigt(x, *p_peaks, sigma_func_left, sigma_func_right) + _quadratic(x, *p_bg)
    else:
        raise ValueError(f"Unknown peak model: {peak_model}")

def fit_peaks_with_quadratic_bg(sline, is_reference_mode=False, peak_model='lorentzian', sigma_func_left=None, sigma_func_right=None):
    px = np.arange(sline.shape[0])

    p_guess_bg = _fit_background_quadratic(px, sline)
    sline_minus_bg = sline - _quadratic(px, *p_guess_bg)

    pk_ind, pk_info = find_peak_locations(sline_minus_bg, is_reference_mode=is_reference_mode)
    if len(pk_ind) < 1:
        return FittedSpectrum(is_success=False, sline=sline, x_pixels=px)

    pk_ind, pk_info = select_top_two_peaks(pk_ind, pk_info)
    pk_wids = 0.5 * pk_info['widths']
    pk_hts = np.pi * pk_wids * pk_info['peak_heights']

    if len(pk_ind) == 1:
        offset = max(int(0.02 * len(sline)), 1)
        pk_ind = np.array([pk_ind[0] - offset, pk_ind[0] + offset])
        pk_wids = np.array([pk_wids[0], pk_wids[0]])
        pk_hts = np.array([pk_hts[0] / 2, pk_hts[0] / 2])

    p0 = (
        pk_hts[0], pk_ind[0], pk_wids[0],
        pk_hts[1], pk_ind[1], pk_wids[1],
        0,  # offset placeholder
        *p_guess_bg
    )

    lower_bounds = [0, 0, 0, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf]
    upper_bounds = [np.inf, max(px), 20, np.inf, max(px), 20, np.inf, np.inf, np.inf, np.inf]

    fit_func = lambda x, *params: _spectrum_function_quadratic_bg(x, peak_model, sigma_func_left, sigma_func_right, *params)

    popt, _ = curve_fit(
        fit_func,
        px,
        sline,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000
    )
    popt[:7] = sort_peaks(popt[:7])

    return popt

def get_fitted_spectrum_quadratic_bg(sline, is_reference_mode=False, peak_model='lorentzian', sigma_func_left=None, sigma_func_right=None):
    px = np.arange(sline.shape[0])

    popt = fit_peaks_with_quadratic_bg(sline, is_reference_mode, peak_model, sigma_func_left, sigma_func_right)
    fit_func = lambda x, *params: _spectrum_function_quadratic_bg(x, peak_model, sigma_func_left, sigma_func_right, *params)

    fitted_spectrum = fit_func(px, *popt)
    x_fit, y_fit = refine_fitted_spectrum(fit_func, px, popt, factor=10)

    amp1, cen1, wid1, amp2, cen2, wid2 = popt[:6]
    inter_peak_distance = abs(cen2 - cen1)

    return FittedSpectrum(
        is_success=True,
        sline=sline,
        x_pixels=px,
        fitted_spectrum=fitted_spectrum,
        x_fit_refined=x_fit,
        y_fit_refined=y_fit,
        parameters=popt,
        left_peak_center_px=float(cen1),
        left_peak_width_px=float(wid1),
        left_peak_amplitude=float(amp1),
        right_peak_center_px=float(cen2),
        right_peak_width_px=float(wid2),
        right_peak_amplitude=float(amp2),
        inter_peak_distance=inter_peak_distance
    )
