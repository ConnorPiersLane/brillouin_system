from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Callable, Deque, Tuple
from collections import deque

import numpy as np

from brillouin_system.devices.ni.ni_dummy import NIBase
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig

log = get_logger(__name__)


@dataclass
class ReflectionFindingResult:
    found: bool
    z_um: float | None


class ReflectionFinderNI:
    """
    Reflection plane finding while slewing (DAQ and Zaber not hardware-synced).

    Detection target:
      - Define the "reflection interval" as contiguous time where signal > threshold_high.
      - Define interval end using a "return-to-baseline" condition: signal <= threshold_low (typically 0)
        for a debounce duration.
      - Return the midpoint of the above-threshold interval:
            idx_mid = (idx_first + idx_last) // 2

    Block-safe:
      - DAQ returns arbitrary-sized blocks (read_available_block()).
      - We maintain a global sample index counter so intervals spanning multiple blocks are handled.

    Timestamp alignment:
      - For each DAQ read, we take perf_counter() as an anchor for the last sample in that block.
      - Convert idx_mid -> perf time using sample rate.
      - Build z(t) from Zaber position stamps (perf_counter, z) and line-fit; evaluate z at t_mid.

    Notes:
      - This does not require "DAQ start == motion start" alignment.
      - It relies on: stable sample rate, and approximately constant stage velocity over the event window.
    """

    # ----------------------------
    # refine helpers (unchanged)
    # ----------------------------

    @staticmethod
    def parabola_peak_3pt(zs, vals) -> float:
        """
        Sub-sample peak estimate using max point and immediate neighbors.
        """
        zs = np.asarray(zs)
        vals = np.asarray(vals)

        i = int(np.argmax(vals))

        if i == 0 or i == len(vals) - 1:
            return float(zs[i])

        z1, z2, z3 = zs[i - 1], zs[i], zs[i + 1]
        v1, v2, v3 = vals[i - 1], vals[i], vals[i + 1]

        d = z2 - z1
        denom = (v1 - 2 * v2 + v3)
        if denom >= 0:
            return float(z2)

        dz = 0.5 * d * (v1 - v3) / denom
        dz = float(np.clip(dz, -d, d))
        return float(z2 + dz)

    def __init__(
        self,
        daq: NIBase,
        zaber_axis: ZaberEyeLens,
        *,
        scanning_config: ScanningConfig,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ):
        self.daq: NIBase = daq
        self.zaber_lens: ZaberEyeLens = zaber_axis
        self.cancel_cb: Optional[Callable[[], bool]] = cancel_cb

        # parameters slew
        self._n_sigma: int = scanning_config.n_sigma
        self._speed_um_s: float = scanning_config.speed_um_s
        if self._speed_um_s == 0:
            raise ValueError("scan_speed must be nonzero")
        self._max_search_distance_um: float = scanning_config.max_search_distance_um
        self._n_bg_samples: int = self.acquisition_time_to_samples(scanning_config.background_acquisition_time_ms)
        self._backstep_after_search_um: float = scanning_config.backstep_after_search_um

        # parameters refine
        self._do_refine = scanning_config.do_refine
        self._n_point_samples: int = self.acquisition_time_to_samples(scanning_config.point_acquisition_time_ms)
        self._step_um = scanning_config.step_um
        self._range_um = scanning_config.range_um
        self._n_max_values: int = scanning_config.n_max_values

        # --- Slew detection tuning (fast defaults; tweak if needed) ---
        self._threshold_low_default: float = 0.0

        # debounce below threshold_low (ms). prevents flicker ending event early.
        self._debounce_ms: float = 2.0

        # z(t) stamping during slew
        self._z_stamp_period_ms: float = 5.0
        self._fit_window_ms: float = 500.0

    # ----------------------------
    # DAQ utils
    # ----------------------------

    def acquisition_time_to_samples(self, acq_time_ms: float) -> int:
        n = int(round(acq_time_ms * 1e-3 * self.daq.get_sample_rate()))
        return max(1, n)

    def max_values_mean(self, xs) -> float:
        xs = np.asarray(xs)
        s = xs.size
        n = self._n_max_values
        if s == 0:
            raise ValueError("Input Array has size 0")
        if s <= n:
            return float(xs.mean())
        return float(np.partition(xs, s - n)[s - n :].mean())

    # ----------------------------
    # refine scan
    # ----------------------------

    def stop_slewing(self, zaber_lens: ZaberEyeLens):
        zaber_lens.stop_slewing()

    def measure_at(self, z_um: float) -> float:
        self.zaber_lens.move_abs(float(z_um))
        self.daq.flush()
        xs = self.daq.read_block(self._n_point_samples)
        return self.max_values_mean(xs)

    def sample_peak_profile(
        self,
        z_hit: float,
        step_um: float,
        range_um: float,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        n_points = int(range_um / step_um) + 1
        half_span = 0.5 * (n_points - 1) * step_um
        z_start = float(z_hit - half_span)

        zs = np.empty(n_points, dtype=float)
        vals = np.empty(n_points, dtype=float)

        for i in range(n_points):
            if self.cancel_cb and self.cancel_cb():
                log.info("[Reflection Finding] Cancelled during refine stepping.")
                return None, None
            z = z_start + i * step_um
            v = self.measure_at(z)
            zs[i] = z
            vals[i] = v

        return zs, vals

    # ----------------------------
    # z(t) fit helpers
    # ----------------------------

    @staticmethod
    def _linear_fit_z_of_t(stamps: Deque[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        Fit z(t) = a*t + b using least squares over (t,z) stamps.
        Returns (a,b) or None if not enough points / degenerate.
        """
        if len(stamps) < 2:
            return None

        ts = np.fromiter((p[0] for p in stamps), dtype=np.float64, count=len(stamps))
        zs = np.fromiter((p[1] for p in stamps), dtype=np.float64, count=len(stamps))

        t0 = ts.mean()
        z0 = zs.mean()
        dt = ts - t0

        denom = float(np.dot(dt, dt))
        if denom <= 0:
            return None

        a = float(np.dot(dt, zs - z0) / denom)
        b = float(z0 - a * t0)
        return a, b

    @staticmethod
    def _trailing_true_run(mask: np.ndarray) -> int:
        """
        Number of consecutive True values at the END of mask.
        """
        if mask.size == 0:
            return 0
        # Walk backwards until first False
        k = 0
        for v in mask[::-1]:
            if v:
                k += 1
            else:
                break
        return k

    # ----------------------------
    # core slewing scan (MIDPOINT)
    # ----------------------------

    def run_scan_midpoint(
        self,
        *,
        scan_speed: float,
        scan_dist: float,
        threshold_high: float,
        threshold_low: float = 0.0,
    ) -> tuple[bool, float | None]:
        """
        Returns (found, z_mid_um).

        Interval definition:
          - Start: first sample > threshold_high
          - End: after start, once signal is <= threshold_low for debounce duration (consecutive)
          - Midpoint sample index: (first_idx + last_idx) // 2
        """
        fs = float(self.daq.get_sample_rate())
        max_search_time = abs(scan_dist / scan_speed)

        debounce_samps = max(1, int(round((self._debounce_ms * 1e-3) * fs)))

        # motion stamps for z(t)
        stamps: Deque[Tuple[float, float]] = deque()
        fit_window_s = float(self._fit_window_ms * 1e-3)
        z_stamp_period_s = float(self._z_stamp_period_ms * 1e-3)
        next_z_stamp_t = time.perf_counter()

        # global indexing for DAQ samples (block-safe across read_available_block() calls)
        global_next_idx = 0  # next sample index assigned to arr[0] of next block

        # interval tracking
        in_event = False
        first_idx: Optional[int] = None   # global index of first sample > threshold_high
        last_idx: Optional[int] = None    # global index of last sample > threshold_high seen so far

        # counts consecutive samples <= threshold_low *after the last >threshold_high sample*
        below_count = 0

        # perf anchor for sample->time conversion
        last_sample_perf: Optional[float] = None
        last_sample_global_idx: Optional[int] = None

        # start slewing
        self.zaber_lens.start_slewing_guarded(
            speed_um_per_s=float(scan_speed),
            max_distance_um=float(abs(scan_dist)),
        )

        t_start_mono = time.monotonic()
        try:
            while True:
                if self.cancel_cb and self.cancel_cb():
                    log.info("[Reflection Finding] Cancelled during slewing scan.")
                    return False, None

                if (time.monotonic() - t_start_mono) >= max_search_time:
                    return False, None

                # stamp z(t) periodically (perf clock)
                now_perf = time.perf_counter()
                if now_perf >= next_z_stamp_t:
                    try:
                        z_now = float(self.zaber_lens.get_position())
                        stamps.append((now_perf, z_now))

                        # keep only recent
                        t_min = now_perf - fit_window_s
                        while stamps and stamps[0][0] < t_min:
                            stamps.popleft()
                    except Exception:
                        # tolerate occasional read failure
                        pass

                    next_z_stamp_t = now_perf + z_stamp_period_s

                # DAQ read
                samples = self.daq.read_available_block()
                if samples.size == 0:
                    continue

                t_block_end = time.perf_counter()
                n = int(samples.size)
                block_last_idx = global_next_idx + n - 1

                last_sample_perf = t_block_end
                last_sample_global_idx = block_last_idx

                arr = samples  # ndarray

                above = arr > threshold_high
                below_or_eq = arr <= threshold_low

                if not in_event:
                    if np.any(above):
                        # event starts in this block
                        i0 = int(np.argmax(above))  # first True
                        in_event = True
                        first_idx = global_next_idx + i0

                        # last above in this block
                        last_local = int(np.max(np.where(above)[0]))
                        last_idx = global_next_idx + last_local

                        # IMPORTANT: count any below-baseline tail AFTER the last above-threshold sample
                        if last_local < n - 1:
                            tail_mask = below_or_eq[last_local + 1:]
                            below_count = self._trailing_true_run(tail_mask)
                        else:
                            below_count = 0
                else:
                    if np.any(above):
                        # update last above index
                        last_local = int(np.max(np.where(above)[0]))
                        last_idx = global_next_idx + last_local

                        # Restart debounce counting from AFTER the last above-threshold sample
                        if last_local < n - 1:
                            tail_mask = below_or_eq[last_local + 1:]
                            below_count = self._trailing_true_run(tail_mask)
                        else:
                            below_count = 0
                    else:
                        # No above-threshold samples in this block.
                        # Continue counting consecutive <= threshold_low, but only if the sequence is unbroken.
                        if np.all(below_or_eq):
                            below_count += n
                        else:
                            # The run was broken inside this block, so only the trailing run can carry forward.
                            below_count = self._trailing_true_run(below_or_eq)

                    # If we've been below baseline long enough, consider event ended.
                    if below_count >= debounce_samps and first_idx is not None and last_idx is not None:
                        # stop motion ASAP
                        self.zaber_lens.stop_slewing()

                        # midpoint sample index (global)
                        mid_idx = (int(first_idx) + int(last_idx)) // 2

                        # convert mid_idx to perf time using last-sample anchor
                        assert last_sample_perf is not None and last_sample_global_idx is not None
                        samples_from_mid_to_last = int(last_sample_global_idx - mid_idx)
                        if samples_from_mid_to_last < 0:
                            samples_from_mid_to_last = 0

                        t_mid = float(last_sample_perf - (samples_from_mid_to_last / fs))

                        # map t_mid -> z using linear fit of recent stamps (fallback if needed)
                        fit = self._linear_fit_z_of_t(stamps)
                        if fit is not None:
                            a, b = fit
                            z_mid = a * t_mid + b
                        else:
                            # fallback: use instantaneous position and commanded speed
                            pos_now = float(self.zaber_lens.get_position())
                            dt = samples_from_mid_to_last / fs
                            z_mid = pos_now - dt * scan_speed

                        return True, float(z_mid)

                global_next_idx += n

        finally:
            try:
                self.zaber_lens.stop_slewing()
            except Exception:
                pass

    # ----------------------------
    # public entrypoint
    # ----------------------------

    def find_reflection_plane(self, is_go_forwards: bool = True) -> ReflectionFindingResult:
        n_bg_samples = self._n_bg_samples
        n_sigma = self._n_sigma

        speed_um_s = self._speed_um_s if is_go_forwards else (-1.0 * self._speed_um_s)
        max_search_distance_um = self._max_search_distance_um

        with self.daq.streaming():
            self.daq.flush()
            _ = self.daq.read_block(3)

            # background -> threshold_high
            bg: np.ndarray = self.daq.read_block(int(n_bg_samples))
            threshold_high = float(bg.mean() + n_sigma * bg.std())

            found, z_mid = self.run_scan_midpoint(
                scan_speed=speed_um_s,
                scan_dist=max_search_distance_um,
                threshold_high=threshold_high,
                threshold_low=self._threshold_low_default,
            )

            if not found or z_mid is None:
                return ReflectionFindingResult(found=False, z_um=None)

            # backstep after detection
            direction = 1.0 if is_go_forwards else -1.0
            z_mid = float(z_mid - direction * self._backstep_after_search_um)

            if not self._do_refine:
                return ReflectionFindingResult(found=True, z_um=z_mid)

            # Refine (optional)
            zs, vals = self.sample_peak_profile(z_mid, step_um=self._step_um, range_um=self._range_um)

            if zs is None or vals is None:
                return ReflectionFindingResult(found=False, z_um=None)

            imax = int(np.argmax(vals))
            if float(vals[imax]) < threshold_high:
                return ReflectionFindingResult(found=False, z_um=None)

            z_peak2 = self.parabola_peak_3pt(zs, vals)
            return ReflectionFindingResult(found=True, z_um=float(z_peak2))