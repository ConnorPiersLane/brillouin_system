
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FittedSpectrum:
    """
    This fits the classic two lorentzian peak spectrum from Brilloun
    x: pixels
    y: Brillouin signal
    """
    is_success: bool
    x_pixels: np.ndarray # x axis pixels
    sline: np.ndarray # brillouin signal as function of x (summed over y-pixels)
    model: str = ''
    fitted_spectrum: np.ndarray = field(default=None)
    x_fit_refined: np.ndarray = field(default=None)
    y_fit_refined: np.ndarray = field(default=None)
    mask_for_fitting: np.ndarray = field(default=None)
    parameters: np.ndarray = field(default=None)
    left_peak_center_px: float = None
    left_peak_width_px: float = None
    left_peak_amplitude: float = None
    right_peak_center_px: float = None
    right_peak_width_px: float = None
    right_peak_amplitude: float = None
    inter_peak_distance: float = None
    offset: float = None
    # Fitted background level under each peak centre [counts per summed
    # sline pixel]. Feeds the shot-noise term of the precision bound (the
    # pedestal's Poisson noise); None on failed fits and legacy data.
    left_peak_bg_counts: float = None
    right_peak_bg_counts: float = None
    # Outer VIPA orders — filled only by the opt-in 4-peak fit
    # (SpectrumFitter.fit(..., n_peaks=4)); the left/right fields above stay
    # the inner main pair, so downstream consumers never see a difference.
    # None on the production two-peak fit and on legacy data.
    outer_left_peak_center_px: float = None
    outer_left_peak_width_px: float = None
    outer_left_peak_amplitude: float = None
    outer_right_peak_center_px: float = None
    outer_right_peak_width_px: float = None
    outer_right_peak_amplitude: float = None
    outer_left_peak_bg_counts: float = None
    outer_right_peak_bg_counts: float = None

