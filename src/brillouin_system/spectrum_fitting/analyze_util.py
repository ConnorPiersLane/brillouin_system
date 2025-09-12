import numpy as np
import math

from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import sline_from_frame_config


def get_b_values(std_image, fit, k: float = 2.0) -> tuple[float | None, float | None] | None:
    """
    Estimate b (background-noise std per binned pixel) near left/right peaks.

    Background noise is defined here as the standard deviation of the
    background across multiple frames, combined across rows in quadrature,
    then evaluated in a window around each fitted peak.

    Parameters
    ----------
        Provides selected_rows and pixel offsets (same as for sline extraction).
    bg_stats : ImageStatistics
        Contains .std_image (per-pixel std across background frames).
    fit : FittedSpectrum
        Result of spectrum fitting (provides peak centers & widths).
    k : float
        Multiplier for peak width to define the window size around each peak.

    Returns
    -------
    tuple
        (left_b, right_b) where each is the median noise std in the peak window,
        or (None, None) if unavailable.
    """

    # --- Validate inputs ---
    if std_image is None:
        return None
    if not fit.is_success:
        return None, None

    std_img = std_image
    H, W = std_img.shape

    # --- Select rows (same as in get_px_sline_from_image) ---
    sline_config = sline_from_frame_config.get()
    rows = sline_config.selected_rows
    if not rows or not all(0 <= r < H for r in rows):
        print("[get_b_values] Warning: Invalid or empty row list — using full image height.")
        rows = list(range(H))

    # --- Combine noise across rows in quadrature ---
    # Because your signal sums rows, the correct noise combination is:
    # std_sum = sqrt(std1^2 + std2^2 + ...).
    binned_std_full = np.sqrt(np.sum(std_img[rows, :]**2, axis=0))

    # Full detector column axis (absolute pixel positions)
    px_full = np.arange(W)

    # --- Helper: median noise inside a peak window ---
    def side_median_b(center: float, width: float) -> float | None:
        if center is None or width is None:
            return None

        # Convert peak center to int pixel coordinate
        center = int(round(center))

        # Define window around peak: [center - k*width, center + k*width]
        halfwin = int(math.ceil(k * float(width)))
        lo_idx = max(0, center - halfwin)
        hi_idx = min(len(px_full), center + halfwin)

        # Mask pixels inside the peak window
        mask = np.zeros_like(binned_std_full, dtype=bool)
        mask[lo_idx:hi_idx] = True

        if not np.any(mask):
            return None

        # Return median noise level within this window
        return float(np.median(binned_std_full[mask]))

    # --- Apply helper to left/right peaks ---
    left_b  = side_median_b(fit.left_peak_center_px,  fit.left_peak_width_px)
    right_b = side_median_b(fit.right_peak_center_px, fit.right_peak_width_px)

    return left_b, right_b
