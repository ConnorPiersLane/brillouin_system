# ni_reflection_finder_switchable.py
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Literal

import numpy as np

Mode = Literal["midpoint", "max"]


@dataclass
class ReflectionResult:
    found: bool
    mode: Mode
    z_um: Optional[float] = None
    t_event_perf: Optional[float] = None
    idx_first: Optional[int] = None
    idx_last: Optional[int] = None
    idx_event: Optional[int] = None  # midpoint idx or max-idx within interval
    peak_value: Optional[float] = None  # only meaningful in mode="max"


def find_reflection_while_slewing(
    ni,
    zaber,
    *,
    mode: Mode = "midpoint",
    speed_um_per_s: float,
    max_distance_um: float,
    threshold_high: float,
    threshold_low: float,
    debounce_s: float = 0.010,
    poll_s: float = 0.010,
    read_timeout_s: float = 0.010,
    max_time_s: float = 10.0,
    flush_first: bool = True,
    # optional fallback if scan ends before interval ends
    return_best_if_timeout: bool = True,
) -> ReflectionResult:
    """
    Detect a single above-threshold interval using hysteresis + debounce.
    Then return either:
      - mode="midpoint": midpoint index of the interval
      - mode="max": max sample within the interval

    Timestamping:
      - DAQ event time is computed from NI ReadResult.t0_perf + (global_sample_idx / fs)
      - Z is found by interpolating Zaber position log vs perf_counter timestamps

    Requirements from your existing classes:
      - ni.read_available_block(timeout_s=...) -> rr with rr.values, rr.sample_rate_hz, rr.t0_perf
      - ni.flush() (optional)
      - zaber.start_position_log(poll_s=...)
      - zaber.stop_position_log() -> log with .interp(np.array([t]))
      - zaber.start_slewing_guarded(speed_um_per_s, max_distance_um)
      - zaber.stop_slewing()
      - zaber.get_position() (fallback)
    """
    if mode not in ("midpoint", "max"):
        raise ValueError(f"mode must be 'midpoint' or 'max', got {mode!r}")

    if flush_first:
        try:
            ni.flush()
        except Exception:
            pass

    t_start = time.perf_counter()

    # Global sample accounting
    gidx = 0
    t0_global: Optional[float] = None
    fs: Optional[float] = None
    debounce_samples: Optional[int] = None

    # Interval state
    in_interval = False
    idx_first: Optional[int] = None
    idx_last_above: Optional[int] = None
    low_run = 0

    # "max within interval" tracking
    best_v = float("-inf")
    best_idx: Optional[int] = None

    # Keep last-known data in case we timeout while inside an interval
    last_rr_t0: Optional[float] = None  # not strictly needed, but handy for debugging

    zlog = None

    def finalize_with_event_idx(event_idx: int, peak_value: Optional[float]) -> ReflectionResult:
        assert t0_global is not None and fs is not None

        t_event = float(t0_global) + (float(event_idx) / float(fs))

        # stop motion & get log (best effort)
        try:
            zaber.stop_slewing()
        except Exception:
            pass

        nonlocal zlog
        try:
            zlog = zaber.stop_position_log()
        except Exception:
            zlog = None

        if zlog is None or getattr(zlog, "t_perf", np.array([])).size == 0:
            z_event = float(zaber.get_position())
        else:
            z_event = float(zlog.interp(np.array([t_event], dtype=np.float64))[0])

        return ReflectionResult(
            found=True,
            mode=mode,
            z_um=z_event,
            t_event_perf=t_event,
            idx_first=int(idx_first) if idx_first is not None else None,
            idx_last=int(idx_last_above) if idx_last_above is not None else None,
            idx_event=int(event_idx),
            peak_value=float(peak_value) if peak_value is not None else None,
        )

    try:
        zaber.start_position_log(poll_s=poll_s)
        zaber.start_slewing_guarded(float(speed_um_per_s), float(max_distance_um))

        while True:
            if (time.perf_counter() - t_start) > float(max_time_s):
                # Timeout: optionally return best-so-far if we're inside an interval
                if return_best_if_timeout and in_interval and idx_first is not None:
                    if mode == "midpoint":
                        assert idx_last_above is not None
                        event_idx = (idx_first + idx_last_above) // 2
                        return finalize_with_event_idx(event_idx, None)
                    else:
                        if best_idx is None:
                            # interval started but we never updated best (shouldn't happen)
                            best_idx = idx_first
                            best_v_local = None
                        else:
                            best_v_local = best_v
                        return finalize_with_event_idx(int(best_idx), best_v_local)
                return ReflectionResult(found=False, mode=mode)

            rr = ni.read_available_block(timeout_s=float(read_timeout_s))
            arr = rr.values
            n = int(arr.size)
            if n <= 0:
                time.sleep(0.001)
                continue

            last_rr_t0 = float(rr.t0_perf)

            if t0_global is None:
                t0_global = float(rr.t0_perf)
            fs = float(rr.sample_rate_hz)

            if debounce_samples is None:
                debounce_samples = max(1, int(round(float(debounce_s) * fs)))

            th_hi = float(threshold_high)
            th_lo = float(threshold_low)

            for i in range(n):
                v = float(arr[i])
                gi = gidx + i

                if not in_interval:
                    if v > th_hi:
                        in_interval = True
                        idx_first = gi
                        idx_last_above = gi
                        low_run = 0

                        # init max-tracking at interval start
                        best_v = v
                        best_idx = gi
                else:
                    # update last-above and max-tracker
                    if v > th_hi:
                        idx_last_above = gi
                        low_run = 0
                        if mode == "max" and v > best_v:
                            best_v = v
                            best_idx = gi
                    else:
                        # still track max even if between hi/lo (optional but usually desired)
                        if mode == "max" and v > best_v:
                            best_v = v
                            best_idx = gi

                        if v <= th_lo:
                            low_run += 1
                            if low_run >= int(debounce_samples):
                                # Interval ended -> choose event index
                                assert idx_first is not None and idx_last_above is not None
                                if mode == "midpoint":
                                    event_idx = (idx_first + idx_last_above) // 2
                                    return finalize_with_event_idx(event_idx, None)
                                else:
                                    # max within interval
                                    if best_idx is None:
                                        best_idx = (idx_first + idx_last_above) // 2
                                        peak_value = None
                                    else:
                                        peak_value = best_v
                                    return finalize_with_event_idx(int(best_idx), peak_value)
                        else:
                            # between low and high: don't accumulate low_run
                            low_run = 0

            gidx += n

    finally:
        # Safety cleanup: don't leak motion/logging
        try:
            zaber.stop_slewing()
        except Exception:
            pass
        try:
            zaber.stop_position_log()
        except Exception:
            pass


# --- Example usage ---
# from ni6008 import NI6008
# from zaber_eye_lens import ZaberEyeLens
#
# ni = NI6008(sample_rate_hz=1000)
# z = ZaberEyeLens(port="COM5", axis_index=1)
#
# with ni.streaming():
#     res_mid = find_reflection_while_slewing(
#         ni, z,
#         mode="midpoint",
#         speed_um_per_s=200.0,
#         max_distance_um=3000.0,
#         threshold_high=0.8,
#         threshold_low=0.2,
#         debounce_s=0.020,
#         max_time_s=10.0,
#     )
#     res_max = find_reflection_while_slewing(
#         ni, z,
#         mode="max",
#         speed_um_per_s=200.0,
#         max_distance_um=3000.0,
#         threshold_high=0.8,
#         threshold_low=0.2,
#         debounce_s=0.020,
#         max_time_s=10.0,
#     )
# print(res_mid)
# print(res_max)