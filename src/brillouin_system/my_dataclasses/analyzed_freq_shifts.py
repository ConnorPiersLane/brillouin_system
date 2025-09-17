from dataclasses import dataclass


@dataclass
class AnalyzedFreqShifts:
    freq_shift_left_peak_ghz: float | None
    freq_shift_right_peak_ghz: float | None
    freq_shift_peak_distance_ghz: float | None
    fwhm_left_peak_ghz: float | None
    fwhm_right_peak_ghz: float | None
    freq_shift_dc_ghz: float | None = None
    freq_shift_centroid_ghz: float | None = None
