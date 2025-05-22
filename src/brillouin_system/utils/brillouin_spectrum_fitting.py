
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import numpy as np

def find_brillouin_peak_locations(sline):
    PROMINENCE_FRACTION = 0.05
    MIN_PEAK_WIDTH = 2
    MIN_PEAK_HEIGHT = 100
    REL_HEIGHT = 0.5
    WLEN_PIXELS = 30

    pos_sline = np.clip(sline, 0, None)

    pk_ind, pk_info = find_peaks(
        sline,
        prominence=PROMINENCE_FRACTION * np.max(pos_sline),
        width=MIN_PEAK_WIDTH,
        height=MIN_PEAK_HEIGHT,
        rel_height=REL_HEIGHT,
        wlen=WLEN_PIXELS
    )
    return pk_ind, pk_info


def select_top_two_peaks(pk_ind, pk_info):
    if len(pk_ind) <= 2:
        return pk_ind, pk_info

    pk_hts = np.pi * np.asarray(pk_info['widths']) * np.asarray(pk_info['peak_heights'])
    sorted_indices = np.argsort(pk_hts)
    top_two_indices = sorted_indices[-2:]
    selected_pk_ind = pk_ind[top_two_indices]

    selected_pk_info = {}
    for prop, values in pk_info.items():
        selected_pk_info[prop] = np.asarray(values)[top_two_indices]

    return selected_pk_ind, selected_pk_info


def fitSpectrum(sline, xtol=1e-6, ftol=1e-6, maxfev=500):
    pk_ind, pk_info = find_brillouin_peak_locations(sline)
    pix = np.arange(sline.shape[0])

    if len(pk_ind) < 1:
        interPeaksteps = np.nan
        fittedSpect = np.nan * np.ones_like(sline)
        return interPeaksteps, fittedSpect, pix, None

    pk_ind, pk_info = select_top_two_peaks(pk_ind, pk_info)

    pk_wids = 0.5 * pk_info['widths']
    pk_hts = np.pi * pk_wids * pk_info['peak_heights']

    if len(pk_ind) == 1:
        offset = max(int(0.02 * len(sline)), 1)
        pk_ind = np.array([pk_ind[0] - offset, pk_ind[0] + offset])
        pk_wids = np.array([pk_wids[0], pk_wids[0]])
        pk_hts = np.array([pk_hts[0] / 2, pk_hts[0] / 2])

    p0 = [pk_hts[0], pk_ind[0], pk_wids[0],
          pk_hts[1], pk_ind[1], pk_wids[1], np.amin(sline)]

    try:
        popt, _ = curve_fit(_2Lorentzian, pix, sline, p0=p0, ftol=ftol, xtol=xtol)
        interPeaksteps = np.abs(popt[4] - popt[1])
        fittedSpect = _2Lorentzian(pix, *popt)
    except Exception as e:
        print(f"[fitSpectrum] Fitting failed: {e}")
        popt = np.nan
        interPeaksteps = np.nan
        fittedSpect = np.nan * np.ones_like(sline)

    return interPeaksteps, fittedSpect, pix, popt


def _2Lorentzian(x, amp1, cen1, wid1, amp2, cen2, wid2, offs):
    return (amp1 * wid1 ** 2 / ((x - cen1) ** 2 + wid1 ** 2)) + \
           (amp2 * wid2 ** 2 / ((x - cen2) ** 2 + wid2 ** 2)) + offs




def refine_fitted_spectrum(x_pixels: np.ndarray, lorentzian_parameters: list, factor: int):
    """
    Refine the Lorentzian fit by interpolating between x-pixel values with specified density.

    Parameters:
        x_pixels (np.ndarray): Original pixel indices (e.g., np.arange(N))
        lorentzian_parameters (list or np.ndarray): Parameters for _2Lorentzian
        factor (int): Number of interpolated points between each original x step

    Returns:
        x_fit (np.ndarray): Interpolated x-values
        y_fit (np.ndarray): Lorentzian evaluated at x_fit
    """
    x_min = x_pixels.min()
    x_max = x_pixels.max()
    num_points = (len(x_pixels) - 1) * factor + 1

    x_fit = np.linspace(x_min, x_max, num=num_points)
    try:
        y_fit = _2Lorentzian(x_fit, *lorentzian_parameters)
    except Exception as e:
        # print(f"[refine_fitted_spectrum] Fitting failed: {e}")
        y_fit = np.zeros_like(x_fit)  # ✅ match length of x_fit

    return x_fit, y_fit





