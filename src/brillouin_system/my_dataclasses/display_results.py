from dataclasses import dataclass, field

import numpy as np


@dataclass
class DisplayResults:
    is_fitting_available: bool
    frame: np.ndarray
    x_pixels: np.ndarray
    sline: np.ndarray
    mask_for_fitting: np.ndarray = field(default=None)
    x_fit_refined: np.ndarray = field(default=None)
    y_fit_refined: np.ndarray = field(default=None)
    inter_peak_distance: float = None
    freq_shift_ghz: float = None
    hwhm_left_peak: float | None = None
    hwhm_right_peak: float | None = None
    # Instrument-subtracted sample HWHM vs the LAST calibration (GHz).
    # None in reference mode, for non-PSF (plain lorentzian) fits, and
    # when the calibration carries no width model — the frontend blanks
    # the line then.
    linewidth_left_peak: float | None = None
    linewidth_right_peak: float | None = None
    # Per-peak frequencies (GHz) from the calibration tracks, so the live
    # view can show the L-R shift difference (the alignment "lean" meter).
    shift_left_peak: float | None = None
    shift_right_peak: float | None = None
    # True when the shifts above carry the post-hoc NA cone correction
    # (sample mode only; the correction never enters the fit itself).
    na_corrected: bool = False
