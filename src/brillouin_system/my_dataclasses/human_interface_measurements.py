from dataclasses import dataclass

import numpy as np

from brillouin_system.calibration.calibration import (
    AnalyzedFreqShifts,
    CalibrationCalculator,
    CalibrationData,
    CalibrationPolyfitParameters,
    calibrate,
)
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.ccd_characteristics import ccd_config
from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig
from brillouin_system.scan_managers.sweep_scan_config.sweep_scan_config import SweepScanConfig
from brillouin_system.spectrum_fitting.noise_analysis import (
    PixelCountsAndPhotons, TheoreticalPeakStdError, theoretical_precision,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter, normalize_model_name,
    config_requires_reflection_background,
)
from brillouin_system.spectrum_fitting.reflection_background import (
    ReflectionBackground,
    ReflectionBackgroundMapper,
)


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
    # NOTE: an sline_rows field existed 2026-08 only (the acquisition row
    # band) and was REMOVED 2026-08-20 (user decision): the fitter always
    # follows the live config, so storing the band served no purpose. Safe,
    # because calibration and samples share one fitter and hence one band —
    # a common band move is common-mode in pixel space (measured: 2 rows
    # move calibrated shifts 0.1-0.8 MHz per peak, <0.4 MHz on the
    # distance; the ~3-4 MHz/row danger is a cal-vs-sample band MISMATCH,
    # which the shared fitter rules out). Old files carrying the field
    # still load (h5 drops unknown fields, pickles keep it inert).

# -------------- Scan Fitting --------------
@dataclass
class AnalyzedSpectrum:
    fitted_spectrum: FittedSpectrum
    analyzed_shifts: AnalyzedFreqShifts
    photons: PixelCountsAndPhotons
    theoretical_precisions: TheoreticalPeakStdError


# -------------- Functions --------------
def calibration_for_scan(scan: AxialScan, fitter: SpectrumFitter) -> CalibrationCalculator:
    """This scan's own calibration, re-fitted from its raw frames when possible.

    scan.calibration_params was fitted at ACQUISITION time with whatever
    reference model was live then, so it silently pins the peak-centre
    convention of that model. Re-analysing samples with a different lineshape
    against it is the model-mixing trap (~0.27 px, -168 MHz split) that the
    fitter's guard catches between the two live configs but cannot see here.
    Re-fitting the stored frames with the current configs is what keeps the
    calibration and the samples on the same convention.

    Without the raw frames there is nothing to re-fit and no record of which
    model produced the stored polynomial, so a pixel-response re-analysis of
    such a scan is refused rather than quietly mixed.
    """
    sample_model, _ = normalize_model_name(fitter.sample_config.fitting_model)

    if scan.calibration_data is not None:
        degree = (scan.calibration_params.degree
                  if scan.calibration_params is not None
                  else calibration_config.get().degree)
        params = calibrate(data=scan.calibration_data, poyfit_degree=degree,
                           fitter=fitter)
        print(f"[fit_axial_scan] Re-fitted this scan's calibration from its raw "
              f"frames (model={fitter.reference_config.fitting_model}, "
              f"degree={degree}) — shifts may differ from the stored analysis.")
        return CalibrationCalculator(parameters=params)

    if sample_model == "lorentzian_x_psf":
        raise ValueError(
            f"Scan '{scan.id}' carries no raw calibration frames "
            f"(calibration_data is None: recorded before they were stored, or "
            f"with save_calibration_frames off), so its calibration cannot be "
            f"re-fitted and there is no record of the model it was fitted "
            f"with. A PSF-convolved sample fit against a calibration that is "
            f"most likely lorentzian is the -168 MHz mixing trap. Analyse this "
            f"scan with 'lorentzian' instead."
        )

    print("[fit_axial_scan] No raw calibration frames stored — using the "
          "calibration polynomial as fitted at acquisition time.")
    return CalibrationCalculator(parameters=scan.calibration_params)


def fitter_for_scan(scan: AxialScan) -> SpectrumFitter:
    """A SpectrumFitter for re-analyzing a scan — the LIVE config, always.

    USER RULE (2026-08-20): the fitter always does what the current fitter
    config says; nothing stored on the scan steers it (see the note on the
    removed sline_rows field in AxialScan)."""
    return SpectrumFitter()


def fit_axial_scan(scan: AxialScan,
                   fitter: SpectrumFitter | None = None,
                   calibration_calculator: CalibrationCalculator | None = None,
                   ) -> list[AnalyzedSpectrum]:
    """Fit every measurement of a scan against its own calibration.

    fitter / calibration_calculator let a caller that already built them
    (e.g. the analyzer GUI, which also needs the calculator for calibration
    plots) inject them instead of paying for a second calibration re-fit.
    The row band, like everything else, comes from the fitter's LIVE config
    (user rule 2026-08-20 — the stored acquisition band is provenance only);
    calibration and samples share the fitter, so they always share the band.
    """
    spectrum_fitter = fitter if fitter is not None else fitter_for_scan(scan)
    if calibration_calculator is None:
        calibration_calculator = calibration_for_scan(scan, spectrum_fitter)

    is_reference_mode = scan.system_state.is_reference_mode
    rows_used = spectrum_fitter.get_selected_rows(
        np.asarray(scan.measurements[0].frame_andor))

    # The reflection background (prmr preset) needs the packaged
    # reflection template registered onto THIS scan's own calibration —
    # frequency-anchored, so it applies across alignment changes.
    reflection_mapper = None
    if not is_reference_mode and config_requires_reflection_background(
            spectrum_fitter.sample_config):
        reflection_mapper = ReflectionBackgroundMapper(
            ReflectionBackground.load_default(), calibration_calculator,
            n_rows=len(rows_used))

    list_analyzed_spectras: list[AnalyzedSpectrum] = []

    # Frames are fitted RAW (user rule 2026-08-20): nothing is subtracted
    # from the data. The fit's background parameters absorb the dark
    # level, and the Thompson bound removes that level analytically
    # using the ccd_characteristics reference (an electronic offset carries
    # no shot noise). This also matches calibrate(), which always fit the
    # calibration frames raw — sample and reference frames go through the
    # identical treatment.
    for measurement in scan.measurements:
        frame = measurement.frame_andor.copy()

        # Generate sline
        px, sline = spectrum_fitter.get_px_sline_from_image(frame)

        # Fit spectrum
        reflection_bg = (reflection_mapper.render(px)
                       if reflection_mapper is not None else None)
        fitting = spectrum_fitter.fit(px=px, sline=sline, is_reference_mode=is_reference_mode,
                                      reflection_background=reflection_bg)

        analyzed_shift = calibration_calculator.analyze(fitting)

        # Per-peak counts and photons, from the fit parameters alone
        photons = PixelCountsAndPhotons.from_fit(
            fs=fitting,
            preamp_gain=scan.system_state.andor_camera_info.preamp_gain,
            emccd_gain=scan.system_state.andor_camera_info.gain)

        theoretical_std: TheoreticalPeakStdError = theoretical_precision(
            fs=fitting, photons=photons,
            calibration_calculator=calibration_calculator,
            preamp_gain=scan.system_state.andor_camera_info.preamp_gain,
            emccd_gain=scan.system_state.andor_camera_info.gain)

        # Append
        anaylzed_spectra = AnalyzedSpectrum(
            fitted_spectrum=fitting,
            analyzed_shifts=analyzed_shift,
            photons=photons,
            theoretical_precisions=theoretical_std
        )
        list_analyzed_spectras.append(anaylzed_spectra)

    return list_analyzed_spectras


