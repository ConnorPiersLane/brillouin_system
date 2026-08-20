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


def select_top_n_peaks(pk_ind, pk_info, n: int):
    """Select the n strongest peaks by height (amplitude ranking).

    n = 2 keeps the production behaviour (the inner main pair is always
    the brightest); n = 4 keeps the outer VIPA orders as well. Returns
    the selected (indices, properties); fewer than n detected peaks are
    returned as-is.
    """
    if len(pk_ind) <= n:
        return pk_ind, pk_info
    pk_hts = np.asarray(pk_info['peak_heights'])
    top = np.argsort(pk_hts)[-n:]
    return (pk_ind[top],
            {prop: np.asarray(values)[top]
             for prop, values in pk_info.items()})


def select_top_two_peaks(pk_ind, pk_info):
    """Select the two strongest peaks by height (amplitude ranking).

    The inner main pair is always the brightest, so this is the production
    selection; fewer than two detected peaks are returned as-is.
    """
    return select_top_n_peaks(pk_ind, pk_info, 2)


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
