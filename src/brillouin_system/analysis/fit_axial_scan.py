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
from brillouin_system.analysis.thompson_shot_noise_limit import theoretical_precision
from brillouin_system.spectrum_fitting.reflection_background import (
    ReflectionBackground,
    ReflectionBackgroundMapper,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
    config_requires_reflection_background,
)


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

    # Per-peak counts and photons, from the fit parameters alone.
    photons = PixelCountsAndPhotons.from_fit(
        fs=fitting,
        preamp_gain=system_state.andor_camera_info.preamp_gain,
        emccd_gain=system_state.andor_camera_info.gain)

    return AnalyzedSpectrum(
        fitted_spectrum=fitting,
        analyzed_shifts=calibration_calculator.analyze(fitting),
        photons=photons,
        theoretical_precisions=theoretical_precision(
            fs=fitting, photons=photons,
            calibration_calculator=calibration_calculator,
            preamp_gain=system_state.andor_camera_info.preamp_gain,
            emccd_gain=system_state.andor_camera_info.gain),
    )


def _reflection_mapper_if_required(fitter: SpectrumFitter,
                                   calibration_calculator: CalibrationCalculator,
                                   system_state: SystemState,
                                   first_frame: np.ndarray,
                                   ) -> ReflectionBackgroundMapper | None:
    """The mapped reflection template for prmr sample fits, or None.

    The packaged template is registered onto THIS scan's own calibration —
    frequency-anchored, so it applies across alignment changes.
    """
    if system_state.is_reference_mode:
        return None
    if not config_requires_reflection_background(fitter.sample_config):
        return None
    rows = fitter.get_selected_rows(first_frame)
    return ReflectionBackgroundMapper(
        ReflectionBackground.load_default(), calibration_calculator,
        n_rows=len(rows))


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
