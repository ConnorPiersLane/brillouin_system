"""Re-analysis of a stored axial scan: every frame fitted against the scan's
own calibration, converted to GHz, with photon numbers and the Thompson bound.

USER RULE (2026-08-20): the fitter always does what the current fitter config
says; nothing stored on the scan steers it. Calibration and samples share the
fitter, so they always share the row band.
"""
import numpy as np

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    calibration_calculator_for_scan,
)
from brillouin_system.analysis.analyzed_spectrum import AnalyzedSpectrum
from brillouin_system.my_dataclasses.axial_scan import AxialScan
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.analysis.pixel_counts_and_photons import PixelCountsAndPhotons
from brillouin_system.analysis.thompson_shot_noise_limit import (
    TheoreticalPeakStdError,
    theoretical_precision,
)
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.spectrum_fitting.reflection_background import (
    ReflectionBackgroundMapper,
    get_current_background,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
    config_requires_reflection_background,
)

log = get_logger(__name__)


# One-shot flag: "photon calibration unavailable" is reported once per
# process, not once per frame (a scan would repeat it hundreds of times).
_photon_calibration_warned = False


def photons_and_bound(fitting,
                      calibration_calculator: CalibrationCalculator,
                      system_state: SystemState,
                      ):
    """Photon numbers + Thompson bound for one fit — or EMPTY results when
    the camera mode's photon calibration is unavailable (e.g. EM mode with
    the EM sensitivity never measured).

    Fits and GHz shifts need no gain at all; only this photon/noise layer
    does. An uncalibratable mode must therefore degrade these outputs to
    None (shown as N/A) instead of blocking the whole analysis — the loud
    guard stays in electrons_per_count for anyone asking for photon
    numbers directly.
    """
    global _photon_calibration_warned
    try:
        photons = PixelCountsAndPhotons.from_fit(
            fs=fitting,
            preamp_gain=system_state.andor_camera_info.preamp_gain,
            emccd_gain=system_state.andor_camera_info.gain)
        theo = theoretical_precision(
            fs=fitting, photons=photons,
            calibration_calculator=calibration_calculator,
            preamp_gain=system_state.andor_camera_info.preamp_gain,
            emccd_gain=system_state.andor_camera_info.gain)
        return photons, theo
    except ValueError as e:
        if not _photon_calibration_warned:
            _photon_calibration_warned = True
            log.warning(f"[analysis] Photon numbers and Thompson bounds are "
                        f"unavailable for this camera mode — fits and shifts "
                        f"are unaffected. Reported once. Cause: {e}")
        return (PixelCountsAndPhotons(None, None, None, None, None, None),
                TheoreticalPeakStdError())


def analyze_frame(frame: np.ndarray,
                  fitter: SpectrumFitter,
                  calibration_calculator: CalibrationCalculator,
                  system_state: SystemState,
                  reflection_mapper: ReflectionBackgroundMapper | None = None,
                  ) -> AnalyzedSpectrum:
    """ONE frame in, ONE AnalyzedSpectrum out: fit, GHz, photons, bound.

    The frame is fitted RAW (user rule 2026-08-20): nothing is subtracted
    from the data. The fit's background parameters absorb the dark level,
    and the Thompson bound removes that level analytically using the
    ccd_characteristics reference (an electronic offset carries no shot
    noise). This matches calibrate(), which always fits the calibration
    frames raw — sample and reference frames get the identical treatment.
    """
    px, sline = fitter.get_px_sline_from_image(frame)
    reflection_bg = (reflection_mapper.render(px)
                     if reflection_mapper is not None else None)
    fitting = fitter.fit(px=px, sline=sline,
                         is_reference_mode=system_state.is_reference_mode,
                         reflection_background=reflection_bg)

    photons, theo = photons_and_bound(fitting, calibration_calculator,
                                      system_state)

    return AnalyzedSpectrum(
        fitted_spectrum=fitting,
        analyzed_shifts=calibration_calculator.analyze(fitting),
        photons=photons,
        theoretical_precisions=theo,
    )


def _reflection_mapper_if_required(fitter: SpectrumFitter,
                                   calibration_calculator: CalibrationCalculator,
                                   system_state: SystemState,
                                   first_frame: np.ndarray,
                                   ) -> ReflectionBackgroundMapper | None:
    """The mapped reflection template for prmr sample fits, or None.

    The current template (user-selected, no default fallback) is registered
    onto THIS scan's own calibration — frequency-anchored, so it applies
    across alignment changes. With none loaded the fits warn and drop the
    reflection term (per-peak flat offsets only).
    """
    if system_state.is_reference_mode:
        return None
    if not config_requires_reflection_background(fitter.sample_config):
        return None
    background = get_current_background()
    if background is None:
        log.warning("[analysis] The config asks for the 'reflection' "
                    "background but none is loaded — fitting this scan "
                    "WITHOUT the reflection term (per-peak offsets only). "
                    "Load one via the analyzer's 'Load Background'.")
        return None
    rows = fitter.get_selected_rows(first_frame)
    return ReflectionBackgroundMapper(
        background, calibration_calculator,
        rows=rows,
        g_margin_ghz=getattr(fitter.sample_config,
                             "reflection_margin_ghz", None))


def fit_axial_scan(scan: AxialScan,
                   fitter: SpectrumFitter | None = None,
                   calibration_calculator: CalibrationCalculator | None = None,
                   ) -> list[AnalyzedSpectrum]:
    """Fit every measurement of a scan against its own calibration.

    fitter / calibration_calculator let a caller that already built them
    (e.g. the analyzer GUI, which also needs the calculator for calibration
    plots) inject them instead of paying for a second calibration re-fit.
    """
    fitter = fitter if fitter is not None else SpectrumFitter()
    if calibration_calculator is None:
        calibration_calculator = calibration_calculator_for_scan(
            scan.calibration_data, scan.calibration_params, fitter)

    reflection_mapper = _reflection_mapper_if_required(
        fitter, calibration_calculator, scan.system_state,
        np.asarray(scan.measurements[0].frame_andor))

    return [
        analyze_frame(
            frame=measurement.frame_andor.copy(),
            fitter=fitter,
            calibration_calculator=calibration_calculator,
            system_state=scan.system_state,
            reflection_mapper=reflection_mapper,
        )
        for measurement in scan.measurements
    ]
