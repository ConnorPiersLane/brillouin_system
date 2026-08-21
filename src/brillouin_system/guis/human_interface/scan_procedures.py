"""The acquisition procedures of the human interface.

Each procedure is a FUNCTION over the backend: it drives the backend's
devices (camera, lens, microwave, DAQ) and display callbacks, builds the
resulting AxialScan / CalibrationData, and registers it on the backend.
The backend itself stays device state + primitives (snap a frame, move
the lens, find the reflection plane) — the same split as the analysis
side, where fit_axial_scan drives the fitter instead of living in it.
"""
import time

from brillouin_system.calibration.calibration import (
    CalibrationData,
    CalibrationMeasurementPoint,
    MeasurementsPerFreq,
)
from brillouin_system.calibration.config.calibration_config import (
    CalibrationConfig,
    calibration_config,
)
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.my_dataclasses.axial_scan import AxialScan
from brillouin_system.my_dataclasses.measurement_point import MeasurementPoint
from brillouin_system.my_dataclasses.request_axial_step_scan import RequestAxialStepScan
from brillouin_system.my_dataclasses.request_sweep_scan import RequestSweepScan
from brillouin_system.my_dataclasses.sweep_cycle import SweepCycle
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult

log = get_logger(__name__)


def take_axial_step_scan(backend, request_axial_scan: RequestAxialStepScan) -> bool:
    lens_x0 = backend.zaber_eye_lens.get_position()
    all_results = []
    reflection_result_forwards: ReflectionResult | None = None
    reflection_result_backwards: ReflectionResult | None = None

    if backend.is_reference_mode:
        log.info(f"[Axial Scan] Measuring N Times the Reference Signal "
                 f"{request_axial_scan.n_measurements}.")

        for i in range(request_axial_scan.n_measurements):
            log.info(f"[Axial Scan] Frame {i + 1}/{request_axial_scan.n_measurements}")
            if backend.f2b_cancel_callback():
                log.info(f"[Axial Scan] Cancelled during step {i + 1}.")
                return False

            frame = backend.get_andor_frame()

            backend.display_spectrum(frame=frame)

            all_results.append(
                MeasurementPoint(
                    frame_andor=frame,
                    lens_zaber_position=lens_x0,
                    time_stamp=time.perf_counter())
            )

    else:

        dx = request_axial_scan.step_size_um

        log.info(f"[Axial Scan] Starting: {request_axial_scan.n_measurements} steps, "
                 f"step size: {request_axial_scan.step_size_um} µm, "
                 f"ID: {request_axial_scan.id}")

        if request_axial_scan.find_reflection_plane:
            reflection_result_forwards = backend.find_reflection_plane(is_go_forwards=True)
            if reflection_result_forwards.found:
                z_pos = (reflection_result_forwards.event_z_um
                         + reflection_result_forwards.z_offset_um)
                backend.zaber_eye_lens.move_abs(z_pos)
            else:
                backend.zaber_eye_lens.move_abs(lens_x0)
                return False

        for i in range(request_axial_scan.n_measurements):
            if backend.f2b_cancel_callback():
                log.info(f"[Axial Scan] Cancelled during step {i + 1}. "
                         f"Returning lens to starting position.")
                backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
                return False

            log.info(f"[Axial Scan] Frame {i + 1}/{request_axial_scan.n_measurements}")
            backend.zaber_eye_lens.move_rel(delta_um=dx)
            zaber_pos = backend.zaber_eye_lens.get_position()
            backend.b2f_emit_update_zaber_lens_position(zaber_pos)

            frame = backend.get_andor_frame()

            backend.display_spectrum(frame=frame)

            all_results.append(
                MeasurementPoint(
                    frame_andor=frame,
                    lens_zaber_position=zaber_pos,
                    time_stamp=time.perf_counter())
            )

    if request_axial_scan.find_reflection_plane:
        reflection_result_backwards = backend.find_reflection_plane(is_go_forwards=False)

    # Move lens back to original position
    backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)

    axial_scan = AxialScan(
        i=backend.next_axial_scan_index(),
        id=request_axial_scan.id,
        measurements=all_results,
        system_state=backend.get_current_system_state(),
        calibration_params=backend.calibration_poly_fit_params,
        calibration_data=backend.calibration_data_to_store(),
        eye_tracker_results=request_axial_scan.eye_tracker_results,
        reflection_result_forwards=reflection_result_forwards,
        reflection_result_backwards=reflection_result_backwards,
    )
    backend.register_axial_scan(axial_scan)

    return True


def _accept_crossing(
    result: ReflectionResult | None,
    *,
    reference_z_um: float,
    gate_um: float,
    reference_peak: float | None,
    min_peak_fraction: float,
    reference_name: str,
) -> tuple[bool, str | None]:
    """
    Decide whether a sweep-scan crossing is the real surface.

    Two independent gates, both needed:
      - DISTANCE from a recent reference. Catches a crossing that is
        plausible in shape but in the wrong place.
      - PEAK AMPLITUDE relative to a reference peak. Every false crossing
        observed on 2026-07-30 was a WEAK peak while genuine ones stayed
        above 0.8x their reference: the plastic cuvette's back wall came in
        at 0.12x, and a finder outlier at 0.006x. Amplitude separates these
        by a wide margin and, unlike distance, needs no trade-off against
        how far the eye may really have moved.

    Returns (accepted, reason_if_rejected).
    """
    if result is None or not result.found:
        return False, "no crossing found"

    delta = result.event_z_um - reference_z_um
    if abs(delta) > gate_um:
        return False, (f"{delta:+.1f} µm from {reference_name} "
                       f"(gate {gate_um:.0f} µm)")

    if reference_peak and result.peak_value is not None:
        frac = result.peak_value / reference_peak
        if frac < min_peak_fraction:
            return False, (f"peak {result.peak_value:.3f} V is {frac:.2f}× the "
                           f"{reference_name} peak {reference_peak:.3f} V "
                           f"(min {min_peak_fraction:.2f}×)")

    return True, None


def take_sweep_scan(backend, request: RequestSweepScan) -> bool:
    """
    In-out sweep scan: repeated find-measure-find cycles.

    One cycle: search inward through the corneal reflection, park at the
    in-crossing + target_depth_um, snap a frame, continue inward
    approach_um past the plane, search outward (recording the
    out-crossing), park approach_um outside the freshest plane estimate,
    turn around. The two crossings of a cycle bracket the frame in time;
    depth labels ((in+out)/2 vs single-crossing) are computed in analysis,
    NOT here — both crossings are stored raw per cycle in sweep_cycles.

    Search speed/detection parameters come from the shared axial
    ScanningConfig; cycle geometry from SweepScanConfig. z_offset_um of
    the plain finder is intentionally NOT applied — the sweep scan's
    target_depth_um replaces it.
    """
    if backend.is_reference_mode:
        log.info("[Sweep Scan] System is in Reference Mode - Change to Sample Mode.")
        return False

    sw = backend.sweep_scan_config
    lens_x0 = backend.zaber_eye_lens.get_position()

    log.info(f"[Sweep Scan] Starting: {sw.n_repeats} cycles, "
             f"target depth {sw.target_depth_um} µm, "
             f"approach {sw.approach_um} µm, ID: {request.id}")

    # Initial full-distance find (normal finder settings) to bootstrap.
    r0: ReflectionResult = backend.find_reflection_plane(is_go_forwards=True)
    if not r0.found:
        log.info("[Sweep Scan] Initial reflection find failed - aborting.")
        backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
        return False
    plane_est = r0.event_z_um
    # Amplitude reference for the in-crossings. The out-crossing of each
    # cycle is instead judged against that cycle's own in-crossing, so a
    # slow legitimate change in signal does not accumulate into a rejection.
    ref_peak = r0.peak_value
    log.info(f"[Sweep Scan] Plane at {plane_est:.1f} µm, reference peak "
             f"{ref_peak:.3f} V (crossings must exceed "
             f"{sw.min_peak_fraction * ref_peak:.3f} V).")

    measurements: list[MeasurementPoint] = []
    cycles: list[SweepCycle] = []

    for k in range(sw.n_repeats):
        if backend.f2b_cancel_callback():
            log.info(f"[Sweep Scan] Cancelled during cycle {k + 1}. "
                     f"Returning lens to starting position.")
            backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
            return False

        log.info(f"[Sweep Scan] Cycle {k + 1}/{sw.n_repeats}")

        # Park outside the current plane estimate and search inward.
        backend.zaber_eye_lens.move_abs(plane_est - sw.approach_um)
        r_in: ReflectionResult = backend.find_reflection_plane(is_go_forwards=True)
        in_ok, in_why = _accept_crossing(
            r_in,
            reference_z_um=plane_est,
            gate_um=sw.plausibility_gate_um,
            reference_peak=ref_peak,
            min_peak_fraction=sw.min_peak_fraction,
            reference_name="the last plane estimate",
        )
        if r_in.found and not in_ok:
            log.warning(f"[Sweep Scan] Cycle {k + 1}: in-crossing at "
                        f"{r_in.event_z_um:.1f} µm rejected - {in_why}.")

        measurement_index = None
        r_out: ReflectionResult | None = None

        if in_ok:
            plane_est = r_in.event_z_um

            # Park at the target depth and take the frame (lens stopped).
            backend.zaber_eye_lens.move_abs(plane_est + sw.target_depth_um)
            time.sleep(sw.settle_s)
            zaber_pos = backend.zaber_eye_lens.get_position()
            backend.b2f_emit_update_zaber_lens_position(zaber_pos)

            frame = backend.get_andor_frame()
            backend.display_spectrum(frame=frame)
            measurements.append(
                MeasurementPoint(
                    frame_andor=frame,
                    lens_zaber_position=zaber_pos,
                    time_stamp=time.perf_counter())
            )
            measurement_index = len(measurements) - 1

            # Continue inward past the plane, then search outward. The
            # out-crossing is judged against THIS cycle's in-crossing —
            # only ~1 s old, so both gates can be tight.
            backend.zaber_eye_lens.move_abs(plane_est + sw.approach_um)
            r_out = backend.find_reflection_plane(is_go_forwards=False)
            out_ok, out_why = _accept_crossing(
                r_out,
                reference_z_um=r_in.event_z_um,
                gate_um=sw.out_gate_um,
                reference_peak=r_in.peak_value,
                min_peak_fraction=sw.min_peak_fraction,
                reference_name="this cycle's in-crossing",
            )
            if r_out.found and not out_ok:
                log.warning(f"[Sweep Scan] Cycle {k + 1}: out-crossing at "
                            f"{r_out.event_z_um:.1f} µm rejected - {out_why}.")
            if out_ok:
                # Freshest estimate for aiming the next cycle. The
                # bias-free (in+out)/2 label is computed in analysis.
                plane_est = r_out.event_z_um
            else:
                log.info(f"[Sweep Scan] Cycle {k + 1}: no valid out-crossing - "
                         f"frame keeps its single-crossing (in) reference.")
        else:
            log.info(f"[Sweep Scan] Cycle {k + 1}: no valid in-crossing - "
                     f"skipping the frame this cycle.")

        cycles.append(SweepCycle(
            cycle_index=k,
            reflection_in=r_in,
            reflection_out=r_out,
            measurement_index=measurement_index,
        ))

    # Park outside the plane, then return the lens to its start position.
    backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)

    n_frames = len(measurements)
    n_pairs = sum(1 for c in cycles
                  if c.measurement_index is not None
                  and c.reflection_out is not None and c.reflection_out.found)
    log.info(f"[Sweep Scan] Done: {n_frames}/{sw.n_repeats} frames taken, "
             f"{n_pairs} with a full in/out pair.")

    axial_scan = AxialScan(
        i=backend.next_axial_scan_index(),
        id=request.id,
        measurements=measurements,
        system_state=backend.get_current_system_state(),
        calibration_params=backend.calibration_poly_fit_params,
        eye_tracker_results=request.eye_tracker_results,
        reflection_result_forwards=r0,
        reflection_result_backwards=None,
        calibration_data=backend.calibration_data_to_store(),
        sweep_cycles=cycles,
        sweep_config=sw,
        scanning_config=backend.axial_scan_config,
    )
    backend.register_axial_scan(axial_scan)

    return n_frames > 0


def perform_calibration(backend) -> bool:
    config: CalibrationConfig = calibration_config.get()

    log.info("[Calibration] Starting calibration.")

    try:
        with backend.force_reference_mode():
            measured_freqs = []

            i = 0
            freqs = config.calibration_freqs
            n = len(freqs)
            for freq in freqs:
                if backend.f2b_cancel_callback():
                    log.info("[Calibration] Cancelled by user.")
                    return False

                backend.microwave.set_frequency(freq)
                i += 1
                log.info(f"Freq {i}/{n}")
                freq_points = []

                for _ in range(config.n_per_freq):
                    if backend.f2b_cancel_callback():
                        log.info("[Calibration] Cancelled by user.")
                        return False

                    frame = backend.get_andor_frame()

                    # The live fit is for the display only; the stored
                    # calibration is raw frames + frequencies — the one
                    # fitting pass happens in calibrate().
                    cali_point = CalibrationMeasurementPoint(
                        frame=frame,
                        microwave_freq=backend.microwave.get_frequency(),
                    )
                    freq_points.append(cali_point)
                    backend.display_spectrum(frame=frame)

                measured_freqs.append(MeasurementsPerFreq(
                    set_freq_ghz=freq,
                    cali_meas_points=freq_points
                ))

            backend.calibration_data = CalibrationData(measured_freqs=measured_freqs)
            backend.update_calibration_calculator()
            log.info(backend.calibration_calculator.get_str_all_models())
            return True

    except Exception as e:
        log.info(f"[Calibration] Exception: {e}")
        return False
