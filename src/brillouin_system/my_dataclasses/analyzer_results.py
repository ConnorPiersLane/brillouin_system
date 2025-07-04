from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum

@dataclass
class AnalyzedFrame:
    frame: np.ndarray
    fitted_spectrum: FittedSpectrum
    freq_shift_left_peak_ghz: float | None
    freq_shift_right_peak_ghz: float | None
    freq_shift_peak_distance_ghz: float | None
    fwhm_left_peak_ghz: float | None
    fwhm_right_peak_ghz: float | None

@dataclass
class PhotonsCounts:
    left_peak_photons: float | None
    right_peak_photons: float | None
    total_photons: float | None

def fitting_to_analyzer_result(frame: np.ndarray,
                               calibration_calculator: CalibrationCalculator,
                               fitting: FittedSpectrum):

    if not fitting.is_success:
        return AnalyzedFrame(
            frame=frame,
            fitted_spectrum=fitting,
            freq_shift_left_peak_ghz=None,
            freq_shift_right_peak_ghz=None,
            freq_shift_peak_distance_ghz=None,
            fwhm_left_peak_ghz=None,
            fwhm_right_peak_ghz=None,
        )
    else:
        return AnalyzedFrame(
            frame=frame,
            fitted_spectrum=fitting,
            freq_shift_left_peak_ghz=calibration_calculator.freq_left_peak(fitting.left_peak_center_px),
            freq_shift_right_peak_ghz=calibration_calculator.freq_right_peak(fitting.right_peak_center_px),
            freq_shift_peak_distance_ghz=calibration_calculator.freq_peak_distance(fitting.inter_peak_distance),
            fwhm_left_peak_ghz=calibration_calculator.df_left_peak(px=fitting.left_peak_center_px, dpx=fitting.left_peak_width_px),
            fwhm_right_peak_ghz=calibration_calculator.df_right_peak(px=fitting.right_peak_center_px, dpx=fitting.right_peak_width_px),
        )


def photon_counts_from_fitted_spectrum(fs: FittedSpectrum) -> PhotonsCounts:
    amp_l = fs.left_peak_amplitude
    amp_r = fs.right_peak_amplitude
    width_l = fs.left_peak_width_px
    width_r = fs.right_peak_width_px
    left_peak_photons = np.pi * amp_l * width_l
    right_peak_photons = np.pi * amp_r * width_r
    return PhotonsCounts(
        left_peak_photons=left_peak_photons,
        right_peak_photons=right_peak_photons,
        total_photons=left_peak_photons + right_peak_photons,
    )

