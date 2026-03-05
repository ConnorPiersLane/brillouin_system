# ni_reflection_finder_simple.py
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np

# Expect these to exist in your project (from your uploaded files):
#   - NI6008.streaming()
#   - NI6008.flush()
#   - NI6008.read_available_block() -> ReadResult(values, sample_rate_hz, t0_perf)
#   - ZaberEyeLens.start_position_log(poll_s=...)
#   - ZaberEyeLens.stop_position_log() -> ZaberPositionLog(t_perf, z_um) with .interp(t_query)
#   - ZaberEyeLens.get_position()
#   - ZaberEyeLens.start_slewing_guarded(speed_um_per_s, max_distance_um)
#   - ZaberEyeLens.stop_slewing()


@dataclass
class ReflectionResult:
    found: bool
    z_um: Optional[float] = None
    t_mid_perf: Optional[float] = None
    idx_first: Optional[int] = None
    idx_last: Optional[int] = None


def find_reflection_while_slewing(
    ni,
    zaber,
    *,
    speed_um_per_s: float,
    max_distance_um: float,
    threshold_high: float,
    threshold_low: float = 0.0,
    debounce_s: float = 0.010,
    poll_s: float = 0.010,
    read_timeout_s: float = 0.010,
    max_time_s: float = 5.0,
    flush_first: bool = True,
) -> ReflectionResult:
    """
    Simple reflection finder:
      - Start Zaber slewing (guarded) + Z position logging (perf_counter timestamps)
      - Continuously read NI blocks
      - Detect a single "above-threshold interval" using hysteresis + debounce
      - Stop, compute midpoint time, interpolate Z at that time, return z_um

    Notes:
      - Zaber and NI are NOT hardware synced. We link them in software by perf_counter.
      - NI ReadResult.t0_perf is a best-effort software anchor.
    """
    t_start = time.perf_counter()

    # Detection state (global sample index since start)
    gidx = 0
    t0_global: Optional[float] = None  # perf_counter time for global sample index 0
    in_interval = False
    idx_first = None
    idx_last_above = None

    # Debounce: require N consecutive samples <= threshold_low to end interval
    debounce_samples = None
    low_run = 0

    # One-time background / stability flush (optional)
    if flush_first:
        try:
            ni.flush()
        except Exception:
            pass

    zlog = None
    try:
        # Start Zaber logging + motion
        zaber.start_position_log(poll_s=poll_s)
        z0 = float(zaber.get_position())
        zaber.start_slewing_guarded(float(speed_um_per_s), float(max_distance_um))

        while True:
            # Timeout
            if (time.perf_counter() - t_start) > float(max_time_s):
                return ReflectionResult(found=False)

            rr = ni.read_available_block(timeout_s=float(read_timeout_s))
            arr = rr.values
            n = int(arr.size)
            fs = float(rr.sample_rate_hz)

            if debounce_samples is None:
                debounce_samples = max(1, int(round(float(debounce_s) * fs)))

            if n <= 0:
                # nothing available right now
                time.sleep(0.001)
                continue

            if t0_global is None:
                # define global sample index 0 time anchor from first received block
                t0_global = float(rr.t0_perf)

            # Process samples in this block
            for i in range(n):
                v = float(arr[i])
                gi = gidx + i  # global index of this sample

                if not in_interval:
                    if v > float(threshold_high):
                        in_interval = True
                        idx_first = gi
                        idx_last_above = gi
                        low_run = 0
                else:
                    # still inside / after interval started
                    if v > float(threshold_high):
                        idx_last_above = gi
                        low_run = 0
                    elif v <= float(threshold_low):
                        low_run += 1
                        if low_run >= debounce_samples:
                            # Interval ended: compute midpoint
                            assert idx_first is not None and idx_last_above is not None
                            idx_mid = (idx_first + idx_last_above) // 2
                            t_mid = float(t0_global) + (idx_mid / fs)

                            # Stop motion, stop logging, interpolate z(t_mid)
                            try:
                                zaber.stop_slewing()
                            except Exception:
                                pass

                            try:
                                zlog = zaber.stop_position_log()
                            except Exception:
                                zlog = None

                            if zlog is None or getattr(zlog, "t_perf", np.array([])).size == 0:
                                # fallback: at least return current position
                                z_mid = float(zaber.get_position())
                            else:
                                z_mid = float(zlog.interp(np.array([t_mid], dtype=np.float64))[0])

                            return ReflectionResult(
                                found=True,
                                z_um=z_mid,
                                t_mid_perf=t_mid,
                                idx_first=int(idx_first),
                                idx_last=int(idx_last_above),
                            )
                    else:
                        # between low and high thresholds: neither above nor fully baseline
                        low_run = 0

            gidx += n

    finally:
        # Safety cleanup
        try:
            zaber.stop_slewing()
        except Exception:
            pass
        try:
            # if caller wants the log they can stop it themselves; but we won't leak a thread
            zaber.stop_position_log()
        except Exception:
            pass


# Example usage:
# from ni6008 import NI6008
# from zaber_eye_lens import ZaberEyeLens
#
# ni = NI6008(sample_rate_hz=2000)
# z = ZaberEyeLens(port="COM5", axis_index=1)
#
# with ni.streaming():
#     res = find_reflection_while_slewing(
#         ni, z,
#         speed_um_per_s=200.0,
#         max_distance_um=2000.0,
#         threshold_high=0.8,
#         threshold_low=0.1,
#         debounce_s=0.010,
#     )
# print(res)