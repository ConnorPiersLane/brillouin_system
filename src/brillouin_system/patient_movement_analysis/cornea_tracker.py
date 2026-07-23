"""
Cornea axial-position sweep tracker (headless, no Qt).

Concept
-------
After the reflection finder has located the corneal front surface, the tracker
oscillates the eye lens +- sweep_amplitude_um around it. Every pass crosses
the surface once; the reflection peak in the DAQ trace is located with the
same validated machinery as the reflection finder (threshold from background
n-sigma, min_samples_above spike rejection, centroid event with sub-sample
resolution, constant-velocity line fit of the Zaber log). Each crossing gives
one surface position measurement with the characterized per-pass precision.

Because consecutive passes travel in opposite directions, the latency-induced
bias (which flips sign with travel direction) cancels in the mean of each
up/down pair: SurfaceEstimate.z_um = (z_up + z_down) / 2 is the bias-free
cornea position. The sweep window optionally re-centers on a moving average
of recent estimates so the tracker follows slow patient drift far beyond the
sweep amplitude.

Blinks / misses: a pass without an accepted reflection interval yields a
TrackPoint with found=False; the window is kept and sweeping continues, so
the tracker reacquires automatically when the reflection returns.

Detection parameters come from the shared axial ScanningConfig (identical to
the GUI reflection finder); sweep parameters come from TrackingConfig.

Usage
-----
    tracker = CorneaTracker(ni, zaber, scan_cfg, track_cfg,
                            on_point=queue.put, on_estimate=queue.put)
    tracker.start(surface_z_um)   # from a prior reflection find
    ...
    points, estimates = tracker.stop()
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_position_log import interp_z_positions
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.patient_movement_analysis.tracking_config.tracking_config import TrackingConfig
from brillouin_system.scan_managers.ni_reflection_finder4 import fit_z_at_time
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """One sweep pass (single crossing attempt)."""
    pass_index: int
    t_perf: float                    # event time if found, else pass end time
    direction: str                   # "up" | "down"
    found: bool
    z_um: Optional[float] = None     # surface z from this pass
    peak_value: Optional[float] = None
    n_samples_above: Optional[int] = None


@dataclass(frozen=True, slots=True)
class SurfaceEstimate:
    """Latency-bias-cancelled estimate from one up/down crossing pair."""
    pair_index: int
    t_perf: float                    # mean of the two event times
    z_um: float                      # (z_up + z_down) / 2
    z_up_um: float
    z_down_um: float


def detect_surface_event(
    values: np.ndarray,
    *,
    background_mean: float,
    threshold_high: float,
    min_samples_above: int,
    centroid_fraction: float,
    gap_samples: int,
) -> tuple[bool, float, int, int, float, int]:
    """
    Locate the reflection event in a complete sweep-pass DAQ trace.

    Samples above threshold_high are clustered (gaps up to gap_samples are
    bridged, mirroring the finder's hysteresis+debounce tolerance); clusters
    with fewer than min_samples_above samples are rejected as noise spikes
    (same rule as the realtime finder). Among the surviving clusters the one
    with the highest peak is taken, and the event position is the centroid of
    samples above background + centroid_fraction * (peak - background) —
    a fractional index, sub-sample resolution.

    Returns (found, frac_index, i_first, i_last, peak_value, n_above).
    """
    v = np.asarray(values, dtype=np.float64)
    above = np.flatnonzero(v > threshold_high)
    if above.size == 0:
        return False, 0.0, 0, 0, 0.0, 0

    # cluster indices, bridging gaps <= gap_samples
    splits = np.flatnonzero(np.diff(above) > gap_samples)
    clusters = np.split(above, splits + 1)

    best = None
    for c in clusters:
        if c.size < min_samples_above:
            continue  # noise spike
        peak = float(np.max(v[c[0]: c[-1] + 1]))
        if best is None or peak > best[0]:
            best = (peak, int(c[0]), int(c[-1]), int(c.size))
    if best is None:
        return False, 0.0, 0, 0, 0.0, 0

    peak, i0, i1, n_above = best
    seg = v[i0: i1 + 1]
    level = background_mean + centroid_fraction * (peak - background_mean)
    w = np.clip(seg - level, 0.0, None)
    w_sum = float(np.sum(w))
    if w_sum > 0.0:
        frac = float(i0) + float(np.sum(w * np.arange(seg.size))) / w_sum
    else:
        frac = float(i0 + np.argmax(seg))
    return True, frac, i0, i1, peak, n_above


class CorneaTracker:
    """
    Sweep tracker thread. Owns NO devices — they are passed in and must not be
    used elsewhere while tracking runs (NI streaming + Zaber motion).
    """

    def __init__(
        self,
        ni,
        zaber,
        scan_cfg: ScanningConfig,
        track_cfg: TrackingConfig,
        *,
        on_point: Optional[Callable[[TrackPoint], None]] = None,
        on_estimate: Optional[Callable[[SurfaceEstimate], None]] = None,
    ):
        self.ni = ni
        self.zaber = zaber
        self.scan_cfg = scan_cfg
        self.track_cfg = track_cfg
        self.on_point = on_point
        self.on_estimate = on_estimate

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

        self.points: list[TrackPoint] = []
        self.estimates: list[SurfaceEstimate] = []

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def start(self, surface_z_um: float) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Tracker already running")
        self._stop_evt.clear()
        self._error = None
        self.points = []
        self.estimates = []
        self._thread = threading.Thread(
            target=self._run, args=(float(surface_z_um),),
            name="CorneaTracker", daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 10.0) -> tuple[list[TrackPoint], list[SurfaceEstimate]]:
        self._stop_evt.set()
        th = self._thread
        if th is not None:
            th.join(timeout=join_timeout_s)
            if th.is_alive():
                raise TimeoutError("CorneaTracker thread did not stop")
        self._thread = None
        return self.points, self.estimates

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_error(self) -> Optional[str]:
        return self._error

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #

    def _run(self, surface_z_um: float) -> None:
        sc = self.scan_cfg
        tc = self.track_cfg
        try:
            self.ni.set_sample_rate_hz(sc.ni_sample_rate_hz)
            fs = self.ni.get_sample_rate_hz()
            gap_samples = max(1, int(round(sc.debounce_s * fs)))

            amp = float(tc.sweep_amplitude_um)
            speed = abs(float(tc.sweep_speed_um_s))
            center = float(surface_z_um)
            recent = deque(maxlen=max(1, int(tc.recenter_avg_points)))

            with self.ni.streaming():
                # Park below the window and measure background there
                # (surface is ~amp away: no reflection in the background).
                self.zaber.move_abs(center - amp)
                self.ni.flush()
                bg = np.asarray(self.ni.read_block(
                    int(tc.bg_acqui_s * fs), timeout_s=tc.bg_acqui_s + 1.0))
                av = float(np.mean(bg))
                sigma = max(float(np.std(bg)), 0.001)
                th_hi = av + sc.threshold_high_n_sigma * sigma
                log.info(f"[Tracker] background {av:.4f} V (sigma {sigma:.4f}), "
                         f"threshold {th_hi:.4f} V, window {center:.1f} +- {amp:.0f} um")

                going_up = True
                pass_index = 0
                pair_index = 0
                last_up: Optional[TrackPoint] = None
                last_down: Optional[TrackPoint] = None
                t_session0 = time.perf_counter()

                while not self._stop_evt.is_set():
                    if (time.perf_counter() - t_session0) > tc.max_track_time_s:
                        log.info("[Tracker] max_track_time_s reached — stopping.")
                        break

                    target = center + amp if going_up else center - amp
                    point = self._sweep_pass(
                        pass_index=pass_index,
                        target_um=target,
                        speed_um_s=speed,
                        av=av,
                        th_hi=th_hi,
                        gap_samples=gap_samples,
                        direction="up" if going_up else "down",
                    )
                    self.points.append(point)
                    if self.on_point is not None:
                        try:
                            self.on_point(point)
                        except Exception:
                            pass

                    if point.found:
                        if point.direction == "up":
                            last_up = point
                        else:
                            last_down = point
                        # a completed consecutive up/down pair -> estimate
                        if last_up is not None and last_down is not None and \
                                abs(last_up.pass_index - last_down.pass_index) == 1:
                            est = SurfaceEstimate(
                                pair_index=pair_index,
                                t_perf=0.5 * (last_up.t_perf + last_down.t_perf),
                                z_um=0.5 * (last_up.z_um + last_down.z_um),
                                z_up_um=last_up.z_um,
                                z_down_um=last_down.z_um,
                            )
                            pair_index += 1
                            self.estimates.append(est)
                            recent.append(est.z_um)
                            if self.on_estimate is not None:
                                try:
                                    self.on_estimate(est)
                                except Exception:
                                    pass
                            if tc.recenter:
                                center = float(np.mean(recent))

                    going_up = not going_up
                    pass_index += 1

        except Exception as e:
            self._error = repr(e)
            log.exception(f"[Tracker] stopped with error: {e}")
        finally:
            try:
                self.zaber.stop_slewing()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # one pass
    # ------------------------------------------------------------------ #

    def _sweep_pass(
        self,
        *,
        pass_index: int,
        target_um: float,
        speed_um_s: float,
        av: float,
        th_hi: float,
        gap_samples: int,
        direction: str,
    ) -> TrackPoint:
        sc = self.scan_cfg
        current = float(self.zaber.get_position())
        delta = target_um - current
        t_pass = abs(delta) / speed_um_s + 0.3

        daq = None
        zlog = None
        try:
            self.ni.flush()
            self.ni.start_acquiring(
                max_sampling_time_s=t_pass + 0.5,
                chunk_size=sc.chunk_size,
                idle_sleep_s=sc.idle_sleep_s,
            )
            self.zaber.start_position_log(poll_s=sc.z_poll_s, alpha=sc.alpha)
            self.zaber.start_slewing_guarded(
                math.copysign(speed_um_s, delta), abs(delta))

            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < t_pass:
                if self._stop_evt.is_set():
                    break
                time.sleep(0.005)
        finally:
            try:
                self.zaber.stop_slewing()
            except Exception:
                pass
            try:
                zlog = self.zaber.stop_position_log()
            except Exception:
                zlog = None
            try:
                daq = self.ni.stop_acquiring()
            except Exception:
                daq = None

        t_end = time.perf_counter()
        if daq is None or zlog is None or zlog.t_perf.size < 2 or daq.values.size == 0:
            return TrackPoint(pass_index=pass_index, t_perf=t_end,
                              direction=direction, found=False)

        found, frac, i0, i1, peak, n_above = detect_surface_event(
            daq.values,
            background_mean=av,
            threshold_high=th_hi,
            min_samples_above=sc.min_samples_above,
            centroid_fraction=0.8,
            gap_samples=gap_samples,
        )
        if not found:
            return TrackPoint(pass_index=pass_index, t_perf=t_end,
                              direction=direction, found=False)

        t_event = daq.time_of(frac)
        zt = np.asarray(zlog.t_perf)
        zz = np.asarray(zlog.z_um)
        z_fit = fit_z_at_time(t_event, zt, zz)
        z_event = z_fit if np.isfinite(z_fit) else interp_z_positions(t_event, zt, zz)
        if not np.isfinite(z_event):
            return TrackPoint(pass_index=pass_index, t_perf=t_end,
                              direction=direction, found=False)

        # Plausibility gate: the lens only travelled current -> target (plus
        # guard-stop overshoot), so a measured event z outside that range is
        # a corrupted estimate (e.g. bad log segment) — count it as a miss
        # rather than poisoning the track and the re-centering average.
        z_lo = min(current, target_um) - 100.0
        z_hi = max(current, target_um) + 100.0
        if not (z_lo <= z_event <= z_hi):
            log.warning(f"[Tracker] pass {pass_index}: event z {z_event:.1f} um "
                        f"outside travel range [{z_lo:.0f}, {z_hi:.0f}] — discarded.")
            return TrackPoint(pass_index=pass_index, t_perf=t_end,
                              direction=direction, found=False)

        return TrackPoint(
            pass_index=pass_index,
            t_perf=float(t_event),
            direction=direction,
            found=True,
            z_um=float(z_event),
            peak_value=float(peak),
            n_samples_above=int(n_above),
        )
