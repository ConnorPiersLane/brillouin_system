from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.camera_settings import AndorCameraSettings, is_conventional_camera_mode
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum

@dataclass
class PhotonsCounts:
    left_peak_photons: float | None
    right_peak_photons: float | None
    total_photons: float | None

@dataclass
class AnalyzedFrame:
    frame: np.ndarray
    fitted_spectrum: FittedSpectrum
    freq_shift_left_peak_ghz: float | None
    freq_shift_right_peak_ghz: float | None
    freq_shift_peak_distance_ghz: float | None
    fwhm_left_peak_ghz: float | None
    fwhm_right_peak_ghz: float | None

    left_peak_photons: float | None
    right_peak_photons: float | None
    total_photons: float | None

def print_analyzed_frame_summary(af: AnalyzedFrame):
    print("AnalyzedFrame Summary:")
    print(f"  freq_shift_left_peak_ghz      : {af.freq_shift_left_peak_ghz}")
    print(f"  freq_shift_right_peak_ghz     : {af.freq_shift_right_peak_ghz}")
    print(f"  freq_shift_peak_distance_ghz  : {af.freq_shift_peak_distance_ghz}")
    print(f"  fwhm_left_peak_ghz            : {af.fwhm_left_peak_ghz}")
    print(f"  fwhm_right_peak_ghz           : {af.fwhm_right_peak_ghz}")
    print(f"  left_peak_photons             : {af.left_peak_photons}")
    print(f"  right_peak_photons            : {af.right_peak_photons}")
    print(f"  total_photons                 : {af.total_photons}")


@dataclass
class AnalyzedFrameStatistics:
    mean_freq_shift_left_peak_ghz: float | None
    std_freq_shift_left_peak_ghz: float | None

    mean_freq_shift_right_peak_ghz: float | None
    std_freq_shift_right_peak_ghz: float | None

    mean_freq_shift_peak_distance_ghz: float | None
    std_freq_shift_peak_distance_ghz: float | None

    mean_fwhm_left_peak_ghz: float | None
    std_fwhm_left_peak_ghz: float | None

    mean_fwhm_right_peak_ghz: float | None
    std_fwhm_right_peak_ghz: float | None

    mean_left_peak_photons: float | None
    std_left_peak_photons: float | None

    mean_right_peak_photons: float | None
    std_right_peak_photons: float | None

    mean_total_photons: float | None
    std_total_photons: float | None





def calculate_photon_counts_from_fitted_spectrum(fs: FittedSpectrum, camera_settings: AndorCameraSettings) -> PhotonsCounts:

    if is_conventional_camera_mode(camera_settings):
        gain_factor = 1
    else:
        gain_factor = camera_settings.emccd_gain
        if gain_factor == 0:
            gain_factor = 1

    preamp_gain = camera_settings.preamp_gain

    count_to_electron_factor = preamp_gain / gain_factor

    amp_l = fs.left_peak_amplitude
    amp_r = fs.right_peak_amplitude
    width_l = fs.left_peak_width_px
    width_r = fs.right_peak_width_px
    left_peak_photons = np.pi * amp_l * width_l
    right_peak_photons = np.pi * amp_r * width_r
    return PhotonsCounts(
        left_peak_photons=left_peak_photons*count_to_electron_factor,
        right_peak_photons=right_peak_photons*count_to_electron_factor,
        total_photons=(left_peak_photons + right_peak_photons)*count_to_electron_factor,
    )

def fitting_to_analyzer_result(frame: np.ndarray,
                               calibration_calculator: CalibrationCalculator,
                               fitting: FittedSpectrum,
                               camera_settings: AndorCameraSettings):

    if not fitting.is_success:
        return AnalyzedFrame(
            frame=frame,
            fitted_spectrum=fitting,
            freq_shift_left_peak_ghz=None,
            freq_shift_right_peak_ghz=None,
            freq_shift_peak_distance_ghz=None,
            fwhm_left_peak_ghz=None,
            fwhm_right_peak_ghz=None,
            left_peak_photons=None,
            right_peak_photons=None,
            total_photons=None,
        )

    else:
        photons_count: PhotonsCounts = calculate_photon_counts_from_fitted_spectrum(fs=fitting,
                                                                                    camera_settings=camera_settings)

        return AnalyzedFrame(
            frame=frame,
            fitted_spectrum=fitting,
            freq_shift_left_peak_ghz=float(calibration_calculator.freq_left_peak(fitting.left_peak_center_px)),
            freq_shift_right_peak_ghz=float(calibration_calculator.freq_right_peak(fitting.right_peak_center_px)),
            freq_shift_peak_distance_ghz=float(calibration_calculator.freq_peak_distance(fitting.inter_peak_distance)),
            fwhm_left_peak_ghz=float(
                abs(calibration_calculator.df_left_peak(fitting.left_peak_center_px, fitting.left_peak_width_px))
            ),
            fwhm_right_peak_ghz=float(
                abs(calibration_calculator.df_right_peak(fitting.right_peak_center_px, fitting.right_peak_width_px))
                    ),
            left_peak_photons=float(photons_count.left_peak_photons),
            right_peak_photons=float(photons_count.right_peak_photons),
            total_photons=float(photons_count.total_photons),
        )


def analyze_frame_statistics(analyzed_frames: list[AnalyzedFrame]) -> AnalyzedFrameStatistics:
    def extract_valid_values(accessor: callable) -> np.ndarray:
        return np.array([
            value for frame in analyzed_frames
            if (value := accessor(frame)) is not None
        ])

    def safe_mean_std(values: np.ndarray) -> tuple[float | None, float | None]:
        if len(values) == 0:
            return None, None
        return float(np.mean(values)), float(np.std(values))

    # Define attribute accessors using lambdas (type-safe, no strings)
    left_freqs = extract_valid_values(lambda f: f.freq_shift_left_peak_ghz)
    right_freqs = extract_valid_values(lambda f: f.freq_shift_right_peak_ghz)
    peak_distances = extract_valid_values(lambda f: f.freq_shift_peak_distance_ghz)
    left_fwhms = extract_valid_values(lambda f: f.fwhm_left_peak_ghz)
    right_fwhms = extract_valid_values(lambda f: f.fwhm_right_peak_ghz)
    left_photons = extract_valid_values(lambda f: f.left_peak_photons)
    right_photons = extract_valid_values(lambda f: f.right_peak_photons)
    total_photons = extract_valid_values(lambda f: f.total_photons)

    # Compute statistics
    mean_lf, std_lf = safe_mean_std(left_freqs)
    mean_rf, std_rf = safe_mean_std(right_freqs)
    mean_pd, std_pd = safe_mean_std(peak_distances)
    mean_fwhm_l, std_fwhm_l = safe_mean_std(left_fwhms)
    mean_fwhm_r, std_fwhm_r = safe_mean_std(right_fwhms)
    mean_lp, std_lp = safe_mean_std(left_photons)
    mean_rp, std_rp = safe_mean_std(right_photons)
    mean_tp, std_tp = safe_mean_std(total_photons)

    return AnalyzedFrameStatistics(
        mean_freq_shift_left_peak_ghz=mean_lf,
        std_freq_shift_left_peak_ghz=std_lf,
        mean_freq_shift_right_peak_ghz=mean_rf,
        std_freq_shift_right_peak_ghz=std_rf,
        mean_freq_shift_peak_distance_ghz=mean_pd,
        std_freq_shift_peak_distance_ghz=std_pd,
        mean_fwhm_left_peak_ghz=mean_fwhm_l,
        std_fwhm_left_peak_ghz=std_fwhm_l,
        mean_fwhm_right_peak_ghz=mean_fwhm_r,
        std_fwhm_right_peak_ghz=std_fwhm_r,
        mean_left_peak_photons=mean_lp,
        std_left_peak_photons=std_lp,
        mean_right_peak_photons=mean_rp,
        std_right_peak_photons=std_rp,
        mean_total_photons=mean_tp,
        std_total_photons=std_tp,
    )
