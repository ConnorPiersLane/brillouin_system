import numpy as np
from scipy.optimize import curve_fit

from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.utils.brillouin_spectrum_fitting import (
    _2Lorentzian,
    find_brillouin_peak_locations,
    select_top_two_peaks
)
from brillouin_system.utils.fit_util import refine_fitted_spectrum, sort_lorentzian_peaks

# Functions

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

def _spectrum_function_quadratic_bg(x, *para):
    p_lorentzian = (*para[:6], 0)  # first 6 parameters + dummy offset
    p_bg = para[6:]
    return _2Lorentzian(x, *p_lorentzian) + _quadratic(x, *p_bg)

def fit_lorentzian_peaks_with_quadratic_bg(sline, is_reference_mode=False):
    px = np.arange(sline.shape[0])

    # --- Fit background ---
    p_guess_bg = _fit_background_quadratic(px, sline)
    sline_minus_bg = sline - _quadratic(px, *p_guess_bg)

    # --- Find peaks ---
    pk_ind, pk_info = find_brillouin_peak_locations(sline_minus_bg, is_reference_mode=is_reference_mode)
    if len(pk_ind) < 1:
        return FittedSpectrum(
            is_success=False,
            sline=sline,
            x_pixels=px,
        )

    # --- Select top 2 peaks ---
    pk_ind, pk_info = select_top_two_peaks(pk_ind, pk_info)
    pk_wids = 0.5 * pk_info['widths']
    pk_hts = np.pi * pk_wids * pk_info['peak_heights']

    if len(pk_ind) == 1:
        offset = max(int(0.02 * len(sline)), 1)
        pk_ind = np.array([pk_ind[0] - offset, pk_ind[0] + offset])
        pk_wids = np.array([pk_wids[0], pk_wids[0]])
        pk_hts = np.array([pk_hts[0] / 2, pk_hts[0] / 2])

    # --- Initial guesses ---
    amp1_guess, cen1_guess, wid1_guess = pk_hts[0], pk_ind[0], pk_wids[0]
    amp2_guess, cen2_guess, wid2_guess = pk_hts[1], pk_ind[1], pk_wids[1]

    p0 = (
        amp1_guess, cen1_guess, wid1_guess,
        amp2_guess, cen2_guess, wid2_guess,
        *p_guess_bg
    )

    # --- Bounds ---
    lower_bounds = [0, 0, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf]
    upper_bounds = [np.inf, max(px), 20, np.inf, max(px), 20, np.inf, np.inf, np.inf]

    # --- Fit full model ---
    popt, _ = curve_fit(
        _spectrum_function_quadratic_bg,
        px,
        sline,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000
    )

    # --- Sort peaks consistently ---
    popt[:7] = sort_lorentzian_peaks(popt[:7])

    return popt

def get_fitted_spectrum_quadratic_bg(sline, is_reference_mode=False):
    popt = fit_lorentzian_peaks_with_quadratic_bg(sline, is_reference_mode=is_reference_mode)
    px = np.arange(sline.shape[0])

    # --- Evaluate fit ---
    fitted_spectrum = _spectrum_function_quadratic_bg(px, *popt)
    x_fit, y_fit = refine_fitted_spectrum(_spectrum_function_quadratic_bg, px, popt, factor=10)

    # --- Unpack Lorentzian parameters ---
    amp1, cen1, wid1, amp2, cen2, wid2 = popt[:6]
    inter_peak_distance = abs(cen2 - cen1)

    return FittedSpectrum(
        is_success=True,
        sline=sline,
        x_pixels=px,
        fitted_spectrum=fitted_spectrum,
        x_fit_refined=x_fit,
        y_fit_refined=y_fit,
        lorentzian_parameters=popt,
        left_peak_center_px=float(cen1),
        left_peak_width_px=float(wid1),
        left_peak_amplitude=float(amp1),
        right_peak_center_px=float(cen2),
        right_peak_width_px=float(wid2),
        right_peak_amplitude=float(amp2),
        inter_peak_distance=inter_peak_distance
    )
