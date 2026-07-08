from dataclasses import dataclass

import numpy as np

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationData,
    CalibrationPolyfitParameters,
)
from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts, \
    calculate_photon_counts_from_fitted_spectrum
from brillouin_system.spectrum_fitting.helpers.subtract_background import subtract_background, subtract_darknoise
from brillouin_system.spectrum_fitting.spectrum_analyzer import AnalyzedFreqShifts, TheoreticalPeakStdError, \
    SpectrumAnalyzer
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


# -------------- Request for Scan --------------
@dataclass
class RequestAxialStepScan:
    id: str
    n_measurements: int
    step_size_um: float
    find_reflection_plane: bool | None = None
    eye_tracker_results: EyeTrackerResults | None = None

@dataclass
class RequestAxialContScan:
    id: str
    speed_um_s: float
    find_reflection_plane: bool | None = None
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
    eye_tracker_results: EyeTrackerResults | None = None
    reflection_result_forwards: ReflectionResult | None = None
    reflection_result_backwards: ReflectionResult | None = None
    # Raw calibration reference frames (for PSF reconstruction / future
    # reprocessing). None for datasets recorded before this field existed.
    calibration_data: CalibrationData | None = None

# -------------- Scan Fitting --------------
@dataclass
class AnalyzedSpectrum:
    fitted_spectrum: FittedSpectrum
    analyzed_shifts: AnalyzedFreqShifts
    photons: PhotonsCounts
    theoretical_precisions: TheoreticalPeakStdError


# -------------- Functions --------------
def fit_axial_scan(scan: AxialScan) -> list[AnalyzedSpectrum]:
    spectrum_fitter = SpectrumFitter()
    calibration_calculator = CalibrationCalculator(parameters=scan.calibration_params)
    spectrum_analyzer = SpectrumAnalyzer(calibration_calculator=calibration_calculator)

    # Elastic-line anchors for the DHO fit; None when the stored calibration
    # cannot provide them (the dho models then refuse to fit).
    anchors = calibration_calculator.elastic_anchors() if scan.calibration_params is not None else None

    do_bg_subtraction = scan.system_state.is_do_bg_subtraction_active

    is_reference_mode = scan.system_state.is_reference_mode

    list_analyzed_spectras: list[AnalyzedSpectrum] = []

    for measurement in scan.measurements:
        frame = measurement.frame_andor.copy()

        if do_bg_subtraction:
            frame = subtract_background(frame=frame, bg_frame=scan.system_state.bg_image)
        else:
            frame = subtract_darknoise(frame=frame, darknoise_frame=scan.system_state.dark_image)

        # Generate sline
        px, sline = spectrum_fitter.get_px_sline_from_image(frame)

        # Fit spectrum
        fitting = spectrum_fitter.fit(px=px, sline=sline, is_reference_mode=is_reference_mode, anchors=anchors)

        analyzed_shift = spectrum_analyzer.analyze_spectrum(fitting=fitting)

        # Photon counts
        photons = calculate_photon_counts_from_fitted_spectrum(fs=fitting,
                                                               preamp_gain=scan.system_state.andor_camera_info.preamp_gain,
                                                               emccd_gain=scan.system_state.andor_camera_info.gain)

        if scan.system_state.is_do_bg_subtraction_active:
            bg_frame_std = scan.system_state.bg_image.std_image
        else:
            bg_frame_std = None

        theoretical_std: TheoreticalPeakStdError = spectrum_analyzer.theoretical_precision(
            fs=fitting, photons=photons, bg_frame_std=bg_frame_std,
            preamp_gain = scan.system_state.andor_camera_info.preamp_gain,
        emccd_gain = scan.system_state.andor_camera_info.gain)

        # Append
        anaylzed_spectra = AnalyzedSpectrum(
            fitted_spectrum=fitting,
            analyzed_shifts=analyzed_shift,
            photons=photons,
            theoretical_precisions=theoretical_std
        )
        list_analyzed_spectras.append(anaylzed_spectra)

    return list_analyzed_spectras


