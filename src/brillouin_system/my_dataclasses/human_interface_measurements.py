from dataclasses import dataclass

import numpy as np

from brillouin_system.calibration.calibration import CalibrationCalculator, CalibrationPolyfitParameters
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults
from brillouin_system.my_dataclasses.analyzed_freq_shifts import AnalyzedFreqShifts
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts, \
    calculate_photon_counts_from_fitted_spectrum
from brillouin_system.spectrum_fitting.helpers.subtract_background import subtract_background, subtract_darknoise
from brillouin_system.spectrum_fitting.spectrum_analyzer import SpectrumAnalyzer
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


# -------------- Request for Scan --------------
@dataclass
class RequestAxialStepScan:
    id: str
    n_measurements: int
    step_size_um: float
    eye_tracker_results: EyeTrackerResults | None = None

@dataclass
class RequestAxialContScan:
    id: str
    speed_um_s: float
    max_distance_um: float
    eye_tracker_results: EyeTrackerResults | None = None
# -------------- Scan Result --------------

@dataclass
class MeasurementPoint:
    frame_andor: np.ndarray  # Original frame, not subtracted
    lens_zaber_position: float
    # frame_left_allied: np.ndarray | None = None
    # frame_right_allied: np.ndarray | None = None
    time_stamp: float | None = None



@dataclass
class AxialScan:
    i: int  # internal tracker
    id: str
    measurements: list[MeasurementPoint]
    system_state: SystemState
    calibration_params: CalibrationPolyfitParameters | None
    scan_speed_um_s: float | None = None
    eye_tracker_results: EyeTrackerResults | None = None

# -------------- Scan Fitting --------------

@dataclass
class FittedAxialScan:
    axial_scan: AxialScan
    fitted_spectras: list[FittedSpectrum]
    fitted_photon_counts: list[PhotonsCounts]

# -------------- Scan Analysis --------------
@dataclass
class AnalysedAxialScan:
    fitted_scan: FittedAxialScan
    freq_shifts: list[AnalyzedFreqShifts]


def get_freq_shift(analyzed_freq_shift: AnalyzedFreqShifts):
    config = calibration_config.get()

    if config.reference == "left":
        return analyzed_freq_shift.freq_shift_left_peak_ghz
    elif config.reference == "right":
        return analyzed_freq_shift.freq_shift_right_peak_ghz
    elif config.reference == "distance":
        return analyzed_freq_shift.freq_shift_peak_distance_ghz
    elif config.reference == "centroid":
        return analyzed_freq_shift.freq_shift_centroid_ghz
    elif config.reference == "dc":
        return analyzed_freq_shift.freq_shift_dc_ghz
    else:
        return None


# -------------- Functions --------------
def fit_axial_scan(scan: AxialScan) -> FittedAxialScan:
    spectrum_fitter = SpectrumFitter()


    do_bg_subtraction = scan.system_state.is_do_bg_subtraction_active

    is_reference_mode = scan.system_state.is_reference_mode

    fitted_measurement_points = []
    photons_points = []

    for measurement in scan.measurements:
        frame = measurement.frame_andor.copy()

        if do_bg_subtraction:
            frame = subtract_background(frame=frame, bg_frame=scan.system_state.bg_image)
        else:
            frame = subtract_darknoise(frame=frame, darknoise_frame=scan.system_state.dark_image)

        # Generate sline
        px, sline = spectrum_fitter.get_px_sline_from_image(frame)

        # Fit spectrum
        fitting = spectrum_fitter.fit(px=px, sline=sline, is_reference_mode=is_reference_mode)

        # Photon counts
        photons = calculate_photon_counts_from_fitted_spectrum(fs=fitting,
                                                               preamp_gain=scan.system_state.andor_camera_info.preamp_gain,
                                                               emccd_gain=scan.system_state.andor_camera_info.gain)

        fitted_measurement_points.append(fitting)
        photons_points.append(photons)

    return FittedAxialScan(
        axial_scan=scan,
        fitted_spectras=fitted_measurement_points,
        fitted_photon_counts=photons_points
    )



def analyze_axial_scan(fitted_scan: FittedAxialScan) -> AnalysedAxialScan:

    analyzed_freqs_shifts = []

    if fitted_scan.axial_scan.calibration_params is None:
        none_shifts = AnalyzedFreqShifts(
            freq_shift_left_peak_ghz=None,
            freq_shift_right_peak_ghz=None,
            freq_shift_peak_distance_ghz=None,
            hwhm_left_peak_ghz=None,
            hwhm_right_peak_ghz=None,
        )

        for _ in fitted_scan.fitted_spectras:
            analyzed_freqs_shifts.append(none_shifts)
    else:
        calibration_calculator = CalibrationCalculator(fitted_scan.axial_scan.calibration_params)

        spectrum_analyzer = SpectrumAnalyzer(calibration_calculator=calibration_calculator)

        analyzed_freqs_shifts = []
        for fitting in fitted_scan.fitted_spectras:
            analyzed_freqs_shifts.append(spectrum_analyzer.analyze_spectrum(fitting))

    return AnalysedAxialScan(
        fitted_scan=fitted_scan,
        freq_shifts=analyzed_freqs_shifts
    )
