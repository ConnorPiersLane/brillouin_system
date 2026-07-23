"""
Backend for the patient-movement GUI.

Owns the non-camera devices (NI DAQ, Zaber eye lens, Zaber rig stage) and the
workflows built on them:

  - reflection plane finding (identical call as the main GUI's HiBackend)
  - cornea sweep tracking (CorneaTracker)
  - laser XY calibration (CalibRigLaserPosition, identical to the main GUI)

The Allied Vision cameras / eye tracker are NOT owned here — the frontend
talks to them through the shared EyeTrackerController (same as the main GUI).

In dummy mode the NI + eye lens are replaced by the simulated pair from
patient_movement_analysis.simulated_devices, whose DAQ signal is physically
coupled to a moving simulated cornea — so reflection finding and tracking are
fully functional at the desk.
"""

from __future__ import annotations

import datetime
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import (
    ZaberHumanInterface, ZaberHumanInterfaceDummy)
from brillouin_system.eye_tracker.calibrate_camera_laser_position.calib_rig_laser_position import (
    CalibRigLaserPosition, LaserOffset)
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.patient_movement_analysis.cornea_tracker import (
    CorneaTracker, SurfaceEstimate, TrackPoint)
from brillouin_system.patient_movement_analysis.simulated_devices import (
    SimNI, SimZaberLens, SimulatedCornea)
from brillouin_system.patient_movement_analysis.tracking_config.tracking_config import (
    TrackingConfig, tracking_config)
from brillouin_system.scan_managers.ni_reflection_finder4 import (
    ReflectionResult, find_reflection_realtime)
from brillouin_system.scan_managers.scanning_config.scanning_config import (
    ScanningConfig, axial_scanning_config)

log = get_logger(__name__)


class PmBackend:

    def __init__(self, use_dummy: bool = True):
        self.use_dummy = use_dummy

        if use_dummy:
            self.sim_cornea = SimulatedCornea(
                base_um=9700.0, amplitude_um=20.0, freq_hz=0.3, drift_um_s=0.5)
            self.zaber_eye_lens = SimZaberLens(start_um=9000.0)
            self.ni = SimNI(self.zaber_eye_lens, self.sim_cornea)
            self.zaber_hi = ZaberHumanInterfaceDummy()
            log.info("[PmBackend] Running with SIMULATED devices (moving cornea at ~9700 um).")
        else:
            from brillouin_system.devices.ni.ni6008 import NI6008
            from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
            self.sim_cornea = None
            self.zaber_eye_lens = ZaberEyeLens()
            self.ni = NI6008()
            self.zaber_hi = ZaberHumanInterface()
            log.info("[PmBackend] Running with REAL devices.")

        self._axial_scan_config: ScanningConfig = axial_scanning_config.get()
        self._tracking_config: TrackingConfig = tracking_config.get()

        self._tracker: Optional[CorneaTracker] = None
        self._cancel_requested = False

    # ------------------------------------------------------------------ #
    # config
    # ------------------------------------------------------------------ #

    def update_axial_config(self, cfg: ScanningConfig) -> None:
        self._axial_scan_config = cfg
        log.info("[PmBackend] Axial scanning config updated.")

    def update_tracking_config(self, cfg: TrackingConfig) -> None:
        self._tracking_config = cfg
        log.info("[PmBackend] Tracking config updated.")

    def get_tracking_config(self) -> TrackingConfig:
        return self._tracking_config

    # ------------------------------------------------------------------ #
    # cancel flag (used by laser calibration)
    # ------------------------------------------------------------------ #

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def reset_cancel(self) -> None:
        self._cancel_requested = False

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    # ------------------------------------------------------------------ #
    # reflection plane
    # ------------------------------------------------------------------ #

    def find_reflection_plane(self, is_go_forwards: bool = True,
                              auto_backup: bool = True) -> ReflectionResult:
        """
        Same parameter mapping as HiBackend.find_reflection_plane.

        auto_backup (forward searches only): first back the lens up so the
        search starts behind the plane. Without this, a repeated find fails
        after the previous find parked the lens at plane + z_offset (past
        the surface).
        """
        if self._tracker is not None and self._tracker.is_running():
            raise RuntimeError("Cannot find reflection plane while tracking is running")

        cfg = self._axial_scan_config
        if auto_backup and is_go_forwards:
            backup_um = min(1000.0, 0.5 * float(cfg.max_distance_um))
            current = float(self.zaber_eye_lens.get_position())
            z_start = max(100.0, current - backup_um)
            log.info(f"[PmBackend] Find: backing up {current - z_start:.0f} um "
                     f"to {z_start:.1f} um before searching forward.")
            self.zaber_eye_lens.move_abs(z_start)

        speed = cfg.speed_um_s if is_go_forwards else -cfg.speed_um_s
        result = find_reflection_realtime(
            ni=self.ni,
            zaber=self.zaber_eye_lens,
            ni_sample_rate_hz=cfg.ni_sample_rate_hz,
            speed_um_s=speed,
            max_distance_um=cfg.max_distance_um,
            threshold_high_n_sigma=cfg.threshold_high_n_sigma,
            threshold_low_n_sigma=cfg.threshold_low_n_sigma,
            bg_acqui_s=cfg.bg_acqui_s,
            debounce_s=cfg.debounce_s,
            z_poll_s=cfg.z_poll_s,
            alpha=cfg.alpha,
            chunk_size=cfg.chunk_size,
            idle_sleep_s=cfg.idle_sleep_s,
            z_offset_um=cfg.z_offset_um,
            min_samples_above=cfg.min_samples_above,
        )
        if result.found:
            log.info(f"[PmBackend] Reflection plane at z = {result.event_z_um:.2f} um "
                     f"(peak {result.peak_value:.2f} V, n_above {result.n_samples_above})")
        else:
            log.warning("[PmBackend] Reflection plane NOT found "
                        f"({result.n_rejected_intervals} noise intervals rejected).")
        return result

    # ------------------------------------------------------------------ #
    # cornea tracking
    # ------------------------------------------------------------------ #

    def start_tracking(
        self,
        surface_z_um: float,
        *,
        on_point: Optional[Callable[[TrackPoint], None]] = None,
        on_estimate: Optional[Callable[[SurfaceEstimate], None]] = None,
    ) -> None:
        if self._tracker is not None and self._tracker.is_running():
            raise RuntimeError("Tracking already running")
        self._tracker = CorneaTracker(
            self.ni, self.zaber_eye_lens,
            self._axial_scan_config, self._tracking_config,
            on_point=on_point, on_estimate=on_estimate,
        )
        self._tracker.start(surface_z_um)
        log.info(f"[PmBackend] Tracking started around z = {surface_z_um:.1f} um, "
                 f"sweep +- {self._tracking_config.sweep_amplitude_um:.0f} um.")

    def stop_tracking(self) -> tuple[list[TrackPoint], list[SurfaceEstimate]]:
        if self._tracker is None:
            return [], []
        points, estimates = self._tracker.stop()
        err = self._tracker.get_error()
        if err:
            log.error(f"[PmBackend] Tracker stopped with error: {err}")
        log.info(f"[PmBackend] Tracking stopped: {len(points)} passes, "
                 f"{len(estimates)} pair estimates.")
        return points, estimates

    def is_tracking(self) -> bool:
        return self._tracker is not None and self._tracker.is_running()

    def get_tracking_error(self) -> Optional[str]:
        return self._tracker.get_error() if self._tracker is not None else None

    # ------------------------------------------------------------------ #
    # hold-still test: static DAQ record at the reflection plane
    # ------------------------------------------------------------------ #

    def record_daq_signal(
        self,
        *,
        duration_s: float,
        z_um: Optional[float] = None,
        on_chunk: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> dict:
        """
        Park the lens at z_um (the found reflection plane) and record the raw
        DAQ signal for duration_s. Blocking — run from a worker thread.

        With the beam at the peak (FWHM ~9 um), the voltage staying high
        means the patient holds still to within microns; dips/dropouts are
        motion. on_chunk(t_rel_s, values) is called with incremental data
        for live plotting.
        """
        if self.is_tracking():
            raise RuntimeError("Cannot record DAQ signal while tracking is running")

        cfg = self._axial_scan_config
        self.ni.set_sample_rate_hz(cfg.ni_sample_rate_hz)
        fs = self.ni.get_sample_rate_hz()

        z_parked = None
        with self.ni.streaming():
            if z_um is not None:
                # The finder's event z carries the single-direction latency
                # bias (~5 um at 2000 um/s) — significant against the ~9 um
                # FWHM peak. A quick static peak-seek parks the beam at the
                # actual maximum before recording.
                z_parked = self._seek_peak_static(float(z_um), fs)

            log.info(f"[DAQ Record] {duration_s:.1f} s at z = "
                     f"{z_parked if z_parked is not None else 'current'} um, "
                     f"fs = {fs:.0f} Hz")
            self.ni.flush()
            self.ni.start_acquiring(
                max_sampling_time_s=duration_s + 0.5,
                chunk_size=cfg.chunk_size,
                idle_sleep_s=cfg.idle_sleep_s,
            )
            last = 0
            t0 = time.perf_counter()
            try:
                while (time.perf_counter() - t0) < duration_s:
                    if stop_event is not None and stop_event.is_set():
                        break
                    rr = self.ni.get_new_block_result(last, copy=True)
                    n = int(rr.values.size)
                    if n > 0:
                        t_rel = (rr.ind0 + np.arange(n, dtype=np.float64)) / fs
                        last = rr.ind0 + n
                        if on_chunk is not None:
                            try:
                                on_chunk(t_rel, np.asarray(rr.values))
                            except Exception:
                                pass
                    time.sleep(0.03)
            finally:
                daq = self.ni.stop_acquiring()

        v = np.asarray(daq.values, dtype=np.float64)
        t = np.arange(v.size, dtype=np.float64) / fs
        log.info(f"[DAQ Record] Done: {v.size} samples, mean {v.mean():.3f} V, "
                 f"std {v.std():.3f} V")
        return {
            "t_s": t,
            "values": v,
            "sample_rate_hz": float(fs),
            "z_um": None if z_um is None else float(z_um),
            "z_parked_um": z_parked,
            "duration_s": float(duration_s),
        }

    def _seek_peak_static(self, center_um: float, fs: float,
                          *, half_range_um: float = 14.0, step_um: float = 2.0) -> float:
        """
        Fine static scan around center_um; park the lens at the reflection
        peak (half-max centroid of the measured profile). Requires an open
        NI streaming session. Returns the parked z.
        """
        targets = np.arange(center_um - half_range_um,
                            center_um + half_range_um + 0.5 * step_um, step_um)
        # backlash preload: approach the scan start from below
        self.zaber_eye_lens.move_abs(float(targets[0]) - 50.0)
        n_per_step = max(5, int(round(0.03 * fs)))

        levels = []
        for z_t in targets:
            self.zaber_eye_lens.move_abs(float(z_t))
            time.sleep(0.02)
            self.ni.flush()
            block = self.ni.read_block(n_per_step, timeout_s=1.0)
            levels.append(float(np.mean(block)))
        v = np.asarray(levels)

        base = float(np.min(v))
        peak = float(np.max(v))
        w = np.clip(v - (base + 0.5 * (peak - base)), 0.0, None)
        if np.sum(w) > 0:
            z_peak = float(np.sum(w * targets) / np.sum(w))
        else:
            z_peak = float(targets[int(np.argmax(v))])

        self.zaber_eye_lens.move_abs(z_peak)
        log.info(f"[DAQ Record] Peak seek: parked at {z_peak:.1f} um "
                 f"({z_peak - center_um:+.1f} vs finder z), peak {peak:.2f} V")
        return z_peak

    # ------------------------------------------------------------------ #
    # laser XY calibration (identical workflow as HiBackend)
    # ------------------------------------------------------------------ #

    def run_laser_xy_calibration(self) -> LaserOffset:
        if self.is_tracking():
            raise RuntimeError("Cannot run laser calibration while tracking is running")

        log.info("[Laser XY Calibration] Starting.")
        self.reset_cancel()
        calib = CalibRigLaserPosition(
            ni=self.ni,
            zaber_eye_lens=self.zaber_eye_lens,
            zaber_hi=self.zaber_hi,
            cancel_callback=self._is_cancelled,
            axial_scan_config=self._axial_scan_config,
        )
        try:
            laser_coord_system = calib.run_calibration()
            log.info(f"[Laser XY Calibration] Done. dx={laser_coord_system.dx:.3f}, "
                     f"dy={laser_coord_system.dy:.3f}, dz={laser_coord_system.dz:.3f}")
            return laser_coord_system
        except Exception as e:
            log.exception(f"[Laser XY Calibration] Failed: {e}")
            raise

    # ------------------------------------------------------------------ #
    # saving
    # ------------------------------------------------------------------ #

    def save_track(
        self,
        path: str | Path,
        points: list[TrackPoint],
        estimates: list[SurfaceEstimate],
        pupil_track: Optional[list[dict]] = None,
        reflection_result: Optional[ReflectionResult] = None,
    ) -> Path:
        """Save one tracking session as human-readable JSON."""
        path = Path(path)

        def _f(x):
            if x is None:
                return None
            x = float(x)
            return x if np.isfinite(x) else None

        data = {
            "meta": {
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "use_dummy": self.use_dummy,
                "axial_scan_config": asdict(self._axial_scan_config),
                "tracking_config": asdict(self._tracking_config),
                "reflection_plane_z_um": _f(reflection_result.event_z_um)
                if reflection_result is not None and reflection_result.found else None,
            },
            "points": [
                {
                    "pass_index": p.pass_index,
                    "t_perf": float(p.t_perf),
                    "direction": p.direction,
                    "found": bool(p.found),
                    "z_um": _f(p.z_um),
                    "peak_value": _f(p.peak_value),
                    "n_samples_above": p.n_samples_above,
                }
                for p in points
            ],
            "estimates": [
                {
                    "pair_index": e.pair_index,
                    "t_perf": float(e.t_perf),
                    "z_um": float(e.z_um),
                    "z_up_um": float(e.z_up_um),
                    "z_down_um": float(e.z_down_um),
                }
                for e in estimates
            ],
            "pupil_track": pupil_track or [],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info(f"[PmBackend] Track saved to {path}")
        return path

    # ------------------------------------------------------------------ #
    # shutdown
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        log.info("[PmBackend] Shutting down devices.")
        try:
            if self.is_tracking():
                self.stop_tracking()
        except Exception:
            pass
        for dev, name in ((self.zaber_eye_lens, "eye lens"), (self.zaber_hi, "rig stage")):
            try:
                dev.close()
            except Exception as e:
                log.warning(f"[PmBackend] Closing {name} failed: {e}")
