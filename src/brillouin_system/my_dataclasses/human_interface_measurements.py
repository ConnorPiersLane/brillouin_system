from dataclasses import dataclass, replace

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
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig
from brillouin_system.scan_managers.sweep_scan_config.sweep_scan_config import SweepScanConfig
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts, \
    calculate_photon_counts_from_fitted_spectrum
from brillouin_system.spectrum_fitting.helpers.subtract_background import subtract_background, subtract_darknoise
from brillouin_system.spectrum_fitting.spectrum_analyzer import AnalyzedFreqShifts, TheoreticalPeakStdError, \
    SpectrumAnalyzer
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter, model_requires_anchors


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


@dataclass
class RequestSweepScan:
    id: str
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
class SweepCycle:
    """One in-out cycle of a sweep scan.

    reflection_in / reflection_out are the raw finder results of the inward and
    outward crossing (biases NOT corrected — keep corrections in analysis; the
    per-direction bias is not a settled constant, see 2026-07-30 alternate-mode
    characterization). measurement_index points into AxialScan.measurements for
    the frame taken during this cycle, or is None if the cycle took no frame
    (missed in-crossing). A found=False / gated-out crossing is stored as-is so
    single-crossing fallback cycles stay identifiable in the saved data.
    """
    cycle_index: int
    reflection_in: ReflectionResult | None = None
    reflection_out: ReflectionResult | None = None
    measurement_index: int | None = None


@dataclass
class AxialScan:
    i: int  # internal tracker
    id: str
    measurements: list[MeasurementPoint]
    system_state: SystemState
    calibration_params: CalibrationPolyfitParameters | None
    # The RAW calibration frames this scan was taken with, so its own
    # calibration can be re-fitted later (e.g. with a different lineshape
    # model). Analyses must use each scan's own calibration — session-level
    # calibration files drift tens of MHz against the scans — and that is only
    # possible if the frames travel with the scan. Controlled by
    # calibration_config.save_calibration_frames; None for datasets recorded
    # before the field existed or with the toggle off.
    calibration_data: CalibrationData | None = None
    eye_tracker_results: EyeTrackerResults | None = None
    reflection_result_forwards: ReflectionResult | None = None
    reflection_result_backwards: ReflectionResult | None = None
    # Set only by the in-out sweep scan: one entry per cycle, in order.
    sweep_cycles: list[SweepCycle] | None = None
    # Acquisition provenance. scanning_config holds the search speed and
    # detection thresholds, which the crossing biases depend on (the ~+5 um
    # direction bias is a 2 mm/s number); sweep_config holds the cycle geometry
    # (approach_um is otherwise only recoverable from the raw Zaber logs).
    sweep_config: SweepScanConfig | None = None
    scanning_config: ScanningConfig | None = None
    # Camera rows summed into the spectral line for this scan. Recorded so a
    # re-fit reproduces the acquisition exactly: which rows are summed shifts
    # the individual peaks by ~3-4 MHz per row (the line is tilted), so a
    # re-analysis on a different band would not reproduce the stored shifts.
    # fit_axial_scan() applies these when present.
    sline_rows: list[int] | None = None

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
    # Re-fit on the rows the scan was acquired with, not on whatever the
    # current config says — a different band would shift the peaks by a few
    # MHz each and the re-fit would not reproduce the stored shifts.
    if scan.sline_rows:
        sline_config = replace(
            spectrum_fitter.sline_config,
            selected_rows=list(scan.sline_rows),
            row_selection="manual",
        )
        spectrum_fitter.update_sline_config(sline_config)
    calibration_calculator = CalibrationCalculator(parameters=scan.calibration_params)
    spectrum_analyzer = SpectrumAnalyzer(calibration_calculator=calibration_calculator)

    do_bg_subtraction = scan.system_state.is_do_bg_subtraction_active

    is_reference_mode = scan.system_state.is_reference_mode

    # Models that anchor peaks at the Rayleigh orders (na_lorentzian*) need the
    # anchors from this scan's calibration; raises if it cannot provide them.
    anchors = None
    if not is_reference_mode and model_requires_anchors(spectrum_fitter.sample_config.fitting_model):
        anchors = calibration_calculator.elastic_anchors()

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
        fitting = spectrum_fitter.fit(px=px, sline=sline, is_reference_mode=is_reference_mode,
                                      anchors=anchors)

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


