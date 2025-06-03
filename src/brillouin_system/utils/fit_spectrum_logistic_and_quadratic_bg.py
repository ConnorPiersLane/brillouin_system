import numpy as np

from scipy.optimize import curve_fit

from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.utils.brillouin_spectrum_fitting import _2Lorentzian, find_brillouin_peak_locations, \
    select_top_two_peaks
from brillouin_system.utils.fit_util import refine_fitted_spectrum, sort_lorentzian_peaks


# Functions
def _logistic_step(x, x0, k):
    """Smooth logistic step function centered at x0, with overflow protection."""
    u = -k * (x - x0)
    u_clipped = np.clip(u, -500, 500)
    return 1 / (1 + np.exp(u_clipped))


def _quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c



def _logistic_step_and_quadratic(x, offset, x01, k1, x02, k2, a, b, c):
    """Combined model: product of two logistic steps times a quadratic, plus offset."""
    return (offset +
            _logistic_step(x, x01, k1) *  # Normalize each step to 1 for shape
            _logistic_step(x, x02, k2) *  # Create a "window" between x01 and x02
            _quadratic(x, a, b, c))


def _get_lower_and_upper_bounds_logistic_step_and_quadratic(px, sline):
    # Bounds:       offset      x01         k1  x02         k2   a         b        c
    lower_bounds = [0,          0,          0,  max(px)/2,  -20, -np.inf, -np.inf, -np.inf]
    upper_bounds = [max(sline), max(px)/2,  20, max(px),    0,   np.inf,  np.inf,  np.inf]
    return lower_bounds, upper_bounds

def fit_background_as_logistic_step_and_quadratic(px, sline):


    offset_guess = min(sline) if min(sline) > 0 else 0
    x01_guess = int(max(px) * 0.1)
    k1_guess = 0
    x02_guess = int(max(px) * 0.9)
    k2_guess = -0
    a_guess = 0.0
    b_guess = 0.0
    c_guess = 0.0

    p0 = [offset_guess, x01_guess, k1_guess, x02_guess, k2_guess, a_guess, b_guess, c_guess]

    lower_bounds, upper_bounds = _get_lower_and_upper_bounds_logistic_step_and_quadratic(px, sline)

    popt, _ = curve_fit(
        _logistic_step_and_quadratic,
        px,
        sline,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000
    )

    return popt

def spectrum_function(x, *para):
    # Split the parameters
    p_lorentzian = (*para[:6], 0)  # first 6 parameters
    p_bg = para[6:]  # the rest

    # Calculate and return the combined fit
    return _2Lorentzian(x, *p_lorentzian) + _logistic_step_and_quadratic(x, *p_bg)

def fit_lorentzian_peaks_with_logistic_step_and_quadratic_bg(sline, is_reference_mode=False):
    px = np.arange(sline.shape[0])

    # --- Fit background ---
    p_guess_bg = fit_background_as_logistic_step_and_quadratic(px, sline)
    sline_minus_bg = sline - _logistic_step_and_quadratic(px, *p_guess_bg)
    sline_minus_bg = np.clip(sline_minus_bg, 0, None)

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
    lower_bounds_bg, upper_bounds_bg = _get_lower_and_upper_bounds_logistic_step_and_quadratic(px, sline)
    lower_bounds = [0, 0, 0, 0, 0, 0] + lower_bounds_bg
    upper_bounds = [np.inf, max(px), 20, np.inf, max(px), 20] + upper_bounds_bg

    # --- Fit full model ---
    popt, _ = curve_fit(
        spectrum_function,
        px,
        sline,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000
    )

    # --- Sort peaks consistently ---
    popt[:7] = sort_lorentzian_peaks(popt[:7])

    return popt

def get_fitted_spectrum_llq(sline, is_reference_mode=False):


    popt = fit_lorentzian_peaks_with_logistic_step_and_quadratic_bg(sline, is_reference_mode=is_reference_mode)
    px = np.arange(sline.shape[0])

    # --- Evaluate fit ---
    fitted_spectrum = spectrum_function(px, *popt)
    x_fit, y_fit = refine_fitted_spectrum(spectrum_function, px, popt, factor=10)

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

