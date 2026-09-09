"""Re-analysis of a stored axial scan: every frame fitted against the scan's
own calibration, converted to GHz, with photon numbers and the Thompson bound.

USER RULE (2026-08-20): the fitter always does what the current fitter config
says; nothing stored on the scan steers it. Calibration and samples share the
fitter, so they always share the row band.
"""
import numpy as np
from dataclasses import replace

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
from brillouin_system.spectrum_fitting.dho import DhoAxes
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
    config_requires_dho_axes,
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
                  dho_axes: DhoAxes | None = None,
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
                         reflection_background=reflection_bg,
                         dho_axes=dho_axes)

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


def _dho_axes_if_required(fitter: SpectrumFitter,
                          calibration_calculator: CalibrationCalculator,
                          system_state: SystemState,
                          calibration_data=None,
                          first_frame=None,
                          ) -> DhoAxes | None:
    """The per-peak calibration axes for 'dho_x_psf' sample fits, or None.

    Unlike the reflection background there is NO degraded fallback: a DHO
    without its frequency tracks and instrument widths is not fittable, so
    a calibration that cannot supply them raises (loudly, before the scan
    loop starts) instead of silently fitting a different model.

    With sample_config.dho_kernel == "measured" the axes also carry the
    instrument kernels stacked from the scan's raw calibration frames at
    the sample peaks' positions (located on first_frame); both inputs are
    then required and their absence raises for the same reason. On the
    template chain the axes carry the whole node table as well, and the
    fitter picks each frame's kernel from it at the found peak position.
    With sline_config.kernel_source == "file" the node table of the lines
    named by kernel_file_lines is the stored one (kernel_file, e.g. a
    401-point fine sweep); the axis, the other kernels and the envelope
    slopes stay the scan's own (epsf.FileKernels).
    """
    if system_state.is_reference_mode:
        return None
    if not config_requires_dho_axes(fitter.sample_config):
        return None
    axes = calibration_calculator.dho_axes()
    if getattr(fitter.sample_config, "dho_kernel", "parametric") != "measured":
        return axes
    if calibration_data is None or first_frame is None:
        raise ValueError(
            "dho_kernel = 'measured' needs the scan's raw calibration "
            "frames and a sample frame to build the instrument kernels."
        )
    from brillouin_system.spectrum_fitting.measured_kernel import (
        measured_kernels_for_frame, sample_peak_positions)
    profiles = getattr(calibration_calculator.p, "template_profiles", None)
    n_peaks = int(fitter.sline_config.n_peaks)
    measured_env = getattr(fitter.sline_config, "envelope_source", "config") == "measured"
    if profiles is None:
        if getattr(fitter.sline_config, "kernel_source", "scan") == "file":
            raise ValueError(
                "kernel_source = 'file' needs centre_method = 'template': "
                "a stored node table carries the template centre "
                "convention, and a parametric axis with a template kernel "
                "is the ~55 MHz convention-mixing trap.")
        # parametric centres, measured shape (inner pair)
        k_left, k_right = measured_kernels_for_frame(
            calibration_data, fitter, first_frame)
        axes = replace(axes, kernel_left=k_left, kernel_right=k_right)
        if measured_env:
            from brillouin_system.spectrum_fitting.envelope import envelope_from_calibration
            env = envelope_from_calibration(calibration_data, fitter)
            positions = sample_peak_positions(fitter, first_frame, n_peaks=n_peaks)
            axes = replace(axes, env_slopes=tuple(env.slope(x) for x in positions))
        return axes
    # template chain: centres, kernels AND envelope from the same calibration
    # (kernel_source = "file": the named lines' kernels from the stored table)
    from brillouin_system.spectrum_fitting.epsf import kernels_for_fit
    profiles = kernels_for_fit(profiles, fitter.sline_config)
    positions = sample_peak_positions(fitter, first_frame, n_peaks=n_peaks)
    names = profiles.names
    if hasattr(profiles, "check"):
        # the quick match check: file profile vs this scan's own at the
        # sample positions, logged once per scan, a warning on a mismatch
        matches = profiles.check(positions)
        text = profiles.report(positions)
        if all(m.ok for m in matches):
            log.info("[kernels] " + text)
        else:
            log.warning("[kernels] MISMATCH between the stored kernel table and "
                        "this scan's calibration (realignment since the fine "
                        "sweep? refresh kernel_file):\n" + text)
    kernels = {nm: profiles.kernel_at(i, positions[i])
               for i, nm in enumerate(names)}
    env_slopes = (tuple(profiles.env_slope(i, positions[i]) for i in range(len(names)))
                  if profiles.envelope is not None else None)
    # the kernel_* / env_slopes fields are the first-frame snapshot; the
    # fitter re-selects per frame from `profiles` (node table)
    return replace(axes,
                   kernel_left=kernels["left"], kernel_right=kernels["right"],
                   kernel_outer_left=kernels.get("outer_left"),
                   kernel_outer_right=kernels.get("outer_right"),
                   env_slopes=env_slopes, profiles=profiles)


def dho_axes_for_fit(fitter, calibration_calculator, calibration_data, frame):
    """The DhoAxes (with measured kernels when configured) for a SAMPLE fit —
    the one construction shared by fit_axial_scan, the live GUI and the
    analyzer."""
    from types import SimpleNamespace
    return _dho_axes_if_required(
        fitter, calibration_calculator, SimpleNamespace(is_reference_mode=False),
        calibration_data=calibration_data, first_frame=frame)


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
    dho_axes = _dho_axes_if_required(
        fitter, calibration_calculator, scan.system_state,
        calibration_data=scan.calibration_data,
        first_frame=np.asarray(scan.measurements[0].frame_andor))

    return [
        analyze_frame(
            frame=measurement.frame_andor.copy(),
            fitter=fitter,
            calibration_calculator=calibration_calculator,
            system_state=scan.system_state,
            reflection_mapper=reflection_mapper,
            dho_axes=dho_axes,
        )
        for measurement in scan.measurements
    ]
