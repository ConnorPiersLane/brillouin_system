import numpy as np
from scipy.signal import find_peaks
from brillouin_system.config.config import find_peaks_reference_config, find_peaks_sample_config, andor_frame_config

def get_sline_from_image(frame: np.ndarray) -> np.ndarray:
    """
    Sum the specified vertical rows in the image to produce the Brillouin sline.
    If the row list is invalid or empty, use the full vertical range.

    Parameters:
        frame (np.ndarray): 2D image from the camera

    Returns:
        Tuple[np.ndarray, np.ndarray]: (px, sline) where px is the pixel axis and sline is the summed spectrum.

    """
    rows = list(andor_frame_config.get().selected_rows)

    height = frame.shape[0]

    if not rows or not all(0 <= r < height for r in rows):
        print("[select_sline_from_image] Warning: Invalid or empty row list — using full image height.")
        rows = list(range(height))

    sline = frame[rows, :].sum(axis=0)

    return sline



def find_peak_locations(sline, is_reference_mode: bool):

    if is_reference_mode:
        find_peak_config = find_peaks_reference_config.get()
    else:
        find_peak_config = find_peaks_sample_config.get()

    if find_peak_config.prominence_fraction is None:
        prominence = None
    else:
        prominence = find_peak_config.prominence_fraction * np.max(sline)


    if find_peak_config.min_peak_height is None:
        min_peak_height = 1
    else:
        min_peak_height = find_peak_config.min_peak_height

    if find_peak_config.min_peak_width is None:
        min_peak_width = 1
    else:
        min_peak_width = find_peak_config.min_peak_width

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

    # # Sort via Area
    # pk_hts = np.pi * np.asarray(pk_info['widths']) * np.asarray(pk_info['peak_heights'])

    # Sort via Amplitutde
    pk_hts = np.asarray(pk_info['peak_heights'])

    sorted_indices = np.argsort(pk_hts)
    top_two_indices = sorted_indices[-2:]
    selected_pk_ind = pk_ind[top_two_indices]

    selected_pk_info = {}
    for prop, values in pk_info.items():
        selected_pk_info[prop] = np.asarray(values)[top_two_indices]

    return selected_pk_ind, selected_pk_info




def refine_fitted_spectrum(function, x_pixels: np.ndarray, parameters: tuple, factor: int):
    """
    Refine the Lorentzian fit by interpolating between x-pixel values with specified density.

    Parameters:
        x_pixels (np.ndarray): Original pixel indices (e.g., np.arange(N))
        parameters (list or np.ndarray): Parameters for function(x, *parameters)
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
        y_fit = function(x_fit, *parameters)
    except Exception as e:
        # print(f"[refine_fitted_spectrum] Fitting failed: {e}")
        y_fit = np.zeros_like(x_fit)  # ✅ match length of x_fit

    return x_fit, y_fit

def sort_peaks(params: np.ndarray | list) -> np.ndarray:
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