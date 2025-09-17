from scipy.signal import find_peaks
import numpy as np

from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FindPeaksConfig





def find_peak_locations(sline: np.ndarray, config: FindPeaksConfig):
    """
    Locate peaks in a 1D Brillouin signal using parameters from config.

    Parameters:
        sline (np.ndarray): 1D Brillouin spectrum
        config (FindPeaksConfig): Configuration for peak detection

    Returns:
        Tuple of (peak_indices, peak_properties) from scipy.signal.find_peaks
    """
    prominence = config.prominence_fraction * np.max(sline)
    min_peak_height = config.min_peak_height or 1
    min_peak_width = config.min_peak_width or 1
    rel_height = config.rel_height if config.rel_height is not None else 0.5
    wlen_pixels = config.wlen_pixels

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
    """
    Select the two strongest peaks based on peak height (not area).

    Parameters:
        pk_ind: Indices of detected peaks
        pk_info: Properties of detected peaks (from find_peaks)

    Returns:
        Tuple of (selected_peak_indices, selected_peak_properties)
    """
    if len(pk_ind) <= 2:
        return pk_ind, pk_info

    pk_hts = np.asarray(pk_info['peak_heights'])
    sorted_indices = np.argsort(pk_hts)
    top_two_indices = sorted_indices[-2:]

    selected_pk_ind = pk_ind[top_two_indices]
    selected_pk_info = {prop: np.asarray(values)[top_two_indices] for prop, values in pk_info.items()}

    return selected_pk_ind, selected_pk_info


def refine_fitted_spectrum(function, x_pixels: np.ndarray, parameters: tuple, factor: int):
    """
    Refine the Lorentzian (or composite) fit by interpolating more densely.

    Parameters:
        function (callable): Fitting function
        x_pixels (np.ndarray): Original pixel indices (e.g., np.arange(N))
        parameters (list or np.ndarray): Parameters for function(x, *parameters)
        factor (int): Number of interpolated points between each original x step

    Returns:
        Tuple (x_fit, y_fit): interpolated x and y values
    """
    x_min = x_pixels.min()
    x_max = x_pixels.max()
    num_points = (len(x_pixels) - 1) * factor + 1

    x_fit = np.linspace(x_min, x_max, num=num_points)
    try:
        y_fit = function(x_fit, *parameters)
    except Exception:
        y_fit = np.zeros_like(x_fit)

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
