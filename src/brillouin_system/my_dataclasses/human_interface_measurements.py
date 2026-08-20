from dataclasses import dataclass, replace

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


def stored_sline_rows(scan: AxialScan) -> list[int] | None:
    """The scan's stored acquisition row band as a plain int list, or None.

    Scans loaded from HDF5 carry sline_rows as a numpy array, whose truth
    value is ambiguous — `if scan.sline_rows:` raises ValueError on them.
    Always go through this helper.
    """
    rows = getattr(scan, "sline_rows", None)
    if rows is None or len(rows) == 0:
        return None
    return [int(r) for r in rows]


def fitter_for_scan(scan: AxialScan) -> SpectrumFitter:
    """A SpectrumFitter pinned to the scan's stored acquisition row band.

    Re-fit on the rows the scan was acquired with, not on whatever the
    current config says — a different band would shift the peaks by a few
    MHz each and the re-fit would not reproduce the stored shifts.
    """
    spectrum_fitter = SpectrumFitter()
    sline_rows = stored_sline_rows(scan)
    if sline_rows is not None:
        sline_config = replace(
            spectrum_fitter.sline_config,
            selected_rows=sline_rows,
            row_selection="manual",
        )
        spectrum_fitter.update_sline_config(sline_config)
    return spectrum_fitter


def fit_axial_scan(scan: AxialScan,
                   fitter: SpectrumFitter | None = None,
                   calibration_calculator: CalibrationCalculator | None = None,
                   ) -> list[AnalyzedSpectrum]:
    """Fit every measurement of a scan against its own calibration.

    fitter / calibration_calculator let a caller that already built them
    (e.g. the analyzer GUI, which also needs the calculator for calibration
    plots) inject them instead of paying for a second calibration re-fit.
    A supplied fitter must carry the scan's row band — build it with
    fitter_for_scan().
    """
    spectrum_fitter = fitter if fitter is not None else fitter_for_scan(scan)
    sline_rows = stored_sline_rows(scan)
    if calibration_calculator is None:
        calibration_calculator = calibration_for_scan(scan, spectrum_fitter)

    is_reference_mode = scan.system_state.is_reference_mode

    # The reflection background (prmr preset) needs the packaged
    # reflection template registered onto THIS scan's own calibration —
    # frequency-anchored, so it applies across alignment changes.
    reflection_mapper = None
    if not is_reference_mode and config_requires_reflection_background(
            spectrum_fitter.sample_config):
        n_rows = len(sline_rows) if sline_rows is not None else None
        reflection_mapper = ReflectionBackgroundMapper(
            ReflectionBackground.load_default(), calibration_calculator,
            n_rows=n_rows)

    list_analyzed_spectras: list[AnalyzedSpectrum] = []

    # Frames are fitted RAW (user rule 2026-08-20): nothing is subtracted
    # from the data. The fit's background parameters absorb the dark/bias
    # pedestal, and the Thompson bound removes that level analytically
    # (an electronic offset carries no shot noise). This also matches
    # calibrate(), which always fit the calibration frames raw — sample and
    # reference frames now go through the identical treatment.
    # Dark/bias level per pixel for the bound: the scan's own dark stack
    # wins; the ccd_characteristics reference value is the fallback; a
    # frame median only if even that is unset.
    dark = scan.system_state.dark_image
    if dark is not None:
        dark_level_per_px = float(np.median(dark.mean_image))
    else:
        dark_level_per_px = ccd_config.get().dark_median_counts or None

    for measurement in scan.measurements:
        frame = measurement.frame_andor.copy()

        # Generate sline
        px, sline = spectrum_fitter.get_px_sline_from_image(frame)

        # Fit spectrum
        measured_bg = (reflection_mapper.render(px)
                       if reflection_mapper is not None else None)
        fitting = spectrum_fitter.fit(px=px, sline=sline, is_reference_mode=is_reference_mode,
                                      measured_background=measured_bg)

        analyzed_shift = calibration_calculator.analyze(fitting)

        # Per-peak counts and photons, from the fit parameters alone
        photons = PixelCountsAndPhotons.from_fit(
            fs=fitting,
            preamp_gain=scan.system_state.andor_camera_info.preamp_gain,
            emccd_gain=scan.system_state.andor_camera_info.gain)

        # The fitted pedestal of a raw-frame fit contains the dark/bias
        # level; pass it (per summed sline pixel) so the bound's pedestal
        # shot-noise term only sees the light part. Source: the scan's own
        # dark stack median, or the frame median when no darks were taken.
        level = (dark_level_per_px if dark_level_per_px is not None
                 else float(np.median(frame)))
        n_rows = len(spectrum_fitter.get_selected_rows(frame))
        bias_counts = level * n_rows
        theoretical_std: TheoreticalPeakStdError = theoretical_precision(
            fs=fitting, photons=photons,
            calibration_calculator=calibration_calculator,
            dark_frame_std=dark.std_image if dark is not None else None,
            preamp_gain=scan.system_state.andor_camera_info.preamp_gain,
            emccd_gain=scan.system_state.andor_camera_info.gain,
            pedestal_bias_counts=bias_counts)

        # Append
        anaylzed_spectra = AnalyzedSpectrum(
            fitted_spectrum=fitting,
            analyzed_shifts=analyzed_shift,
            photons=photons,
            theoretical_precisions=theoretical_std
        )
        list_analyzed_spectras.append(anaylzed_spectra)

    return list_analyzed_spectras


