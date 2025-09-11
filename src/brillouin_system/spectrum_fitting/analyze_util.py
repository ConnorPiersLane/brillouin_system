import math

import numpy as np

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


def get_background_values(bg_frame: np.ndarray, fit: FittedSpectrum, k: float = 2.0) -> dict:
    """
    Estimate background values near the left and right peaks.

    Parameters
    ----------
    bg_frame : np.ndarray
        Raw camera background frame.
    fit : FittedSpectrum
        Fit result containing peak positions/widths.
    k : float, optional
        Factor multiplying the peak width to exclude from averaging.
        Default = 2.0 (i.e. background is taken inside ±2 widths).

    Returns
    -------
    dict
        {
            "left_peak_bg": float,
            "right_peak_bg": float
        }
    """
    spectrum_fitter = SpectrumFitter()
    px, sline = spectrum_fitter.get_px_sline_from_image(bg_frame)

    if not fit.is_success:
        return {"left_peak_bg": None, "right_peak_bg": None}

    results = {}

    for label, center, width in [
        ("left_peak_bg", fit.left_peak_center_px, fit.left_peak_width_px),
        ("right_peak_bg", fit.right_peak_center_px, fit.right_peak_width_px),
    ]:
        if center is None or width is None:
            results[label] = None
            continue

        center = round(center)
        width = math.ceil(k * width)
        # Define exclusion window around the peak
        lo = int(max(0, center - width))
        hi = int(min(len(px), center + width))

        # Take all pixels outside the peak region
        mask = np.zeros_like(sline, dtype=bool)
        mask[lo:hi] = True

        # Background = mean of remaining pixels
        bg_value = float(np.mean(sline[mask])) if np.any(mask) else None
        results[label] = bg_value

    return results