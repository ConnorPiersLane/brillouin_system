
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import numpy as np

from brillouin_system.config.config import reference_config, sample_config, sline_config
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum


def get_sline_from_image(frame: np.ndarray) -> np.ndarray:
    """
    Sum the specified vertical rows in the image to produce the Brillouin sline.
    If the row list is invalid or empty, use the full vertical range.

    Parameters:
        frame (np.ndarray): 2D image from the camera

    Returns:
        np.ndarray: Summed 1D spectrum (sline)
    """
    rows = list(sline_config.get().selected_rows)

    height = frame.shape[0]

    if not rows or not all(0 <= r < height for r in rows):
        print("[select_sline_from_image] Warning: Invalid or empty row list — using full image height.")
        rows = list(range(height))

    sline = frame[rows, :].sum(axis=0)
    return sline



def find_brillouin_peak_locations(sline, is_reference_mode: bool):
    pos_sline = np.clip(sline, 0, None)

    if is_reference_mode:
        find_peak_config = reference_config.get()
    else:
        find_peak_config = sample_config.get()

    if find_peak_config.prominence_fraction is None:
        prominence = None
    else:
        prominence = find_peak_config.prominence_fraction * np.max(pos_sline)

    min_peak_width = find_peak_config.min_peak_width
    min_peak_height = find_peak_config.min_peak_height

    if find_peak_config.rel_height is None:
        rel_height = 0.5
    else:
        rel_height = find_peak_config.rel_height

    wlen_pixels = find_peak_config.wlen_pixels


    pk_ind, pk_info = find_peaks(
        sline,
        prominence=prominence,
        width=min_peak_width,
        height=min_peak_height,
        rel_height=rel_height,
        wlen=wlen_pixels
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

def get_fitted_spectrum_from_image(frame: np.ndarray, is_reference_mode) -> FittedSpectrum:
    sline = get_sline_from_image(frame=frame)
    pk_ind, pk_info = find_brillouin_peak_locations(sline, is_reference_mode=is_reference_mode)
    pix = np.arange(sline.shape[0])

    if len(pk_ind) < 1:
        return FittedSpectrum(
            is_success=False,
            frame=frame,
            sline=sline,
            x_pixels=pix,
        )

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
        popt, _ = curve_fit(_2Lorentzian, pix, sline, p0=p0, ftol=1e-6, xtol=1e-6)

        # Ensure peak 1 is left, peak 2 is right
        popt = sort_lorentzian_peaks(popt)

        interPeaksteps = np.abs(popt[4] - popt[1])
        fittedSpect = _2Lorentzian(pix, *popt)

        x_fit, y_fit = refine_fitted_spectrum(pix, popt, factor=10)

        fitted_spectrum = FittedSpectrum(
            is_success=True,
            frame=frame,
            sline=sline,
            x_pixels=pix,
            fitted_spectrum=fittedSpect,
            x_fit_refined=x_fit,
            y_fit_refined=y_fit,
            lorentzian_parameters=popt,
            left_peak_center_px=float(popt[1]),
            left_peak_width_px=float(popt[2]),
            left_peak_amplitude=float(popt[0]),
            right_peak_center_px=float(popt[4]),
            right_peak_width_px=float(popt[5]),
            right_peak_amplitude=float(popt[3]),
            inter_peak_distance=interPeaksteps
        )

    except Exception as e:
        print(f"[fitSpectrum] Fitting failed: {e}")
        return FittedSpectrum(
            is_success=False,
            frame=frame,
            sline=sline,
            x_pixels=pix,
        )

    return fitted_spectrum


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


def sort_lorentzian_peaks(params: np.ndarray | list) -> np.ndarray:
    """
    Ensure Lorentzian fit parameters are sorted by peak center positions (left-to-right).

    Parameters:
        params (array-like): [amp1, cen1, wid1, amp2, cen2, wid2, offset]

    Returns:
        np.ndarray: Reordered parameters so that the left peak comes first
    """
    amp1, cen1, wid1, amp2, cen2, wid2, offset = params

    if cen1 <= cen2:
        return np.array([amp1, cen1, wid1, amp2, cen2, wid2, offset])
    else:
        return np.array([amp2, cen2, wid2, amp1, cen1, wid1, offset])


