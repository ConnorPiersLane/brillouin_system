# ni_reflection_finder_realtime_ts.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Mode = Literal["midpoint", "max"]


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    found: bool
    mode: Mode
    event_index: Optional[int] = None          # index into daq.values
    event_time_perf: Optional[float] = None    # daq.time_of(event_index)
    event_z_um: Optional[float] = None         # interpolated z at event time
    peak_value: Optional[float] = None         # only meaningful for mode="max"
    idx_first: Optional[int] = None            # interval start (index into daq.values)
    idx_last: Optional[int] = None             # last above threshold_high (index into daq.values)


def _interp_z(t_query: float, t_z: np.ndarray, z_um: np.ndarray) -> float:
    """Linear interpolation with monotonic-time cleanup (like your slew scanner)."""
    t_z = np.asarray(t_z, dtype=np.float64)
    z_um = np.asarray(z_um, dtype=np.float64)
    if t_z.size < 2:
        return float("nan")
    keep = np.concatenate(([True], np.diff(t_z) > 0))
    t_z = t_z[keep]
    z_um = z_um[keep]
    if t_z.size < 2:
        return float("nan")
    return float(np.interp(np.array([t_query], dtype=np.float64), t_z, z_um)[0])


def find_reflection_realtime(
    ni,
    zaber,
    *,
    mode: Mode = "midpoint",
    speed_um_s: float,
    max_distance_um: float,
    threshold_high: float,
    threshold_low: float,
    debounce_s: float = 0.020,     # 20 ms at 1 kHz is a good start
    max_time_s: float = 10.0,
    z_poll_s: float = 0.016,       # ~63 Hz like your calibration; set 0 for "as fast as possible"
    chunk_size: int = 2048,
    pretrigger_flush: bool = True,
    z_offset_um: float = 0.0,      # optional manual offset like your slew scanner
) -> ReflectionResult:
    """
    Early-stop reflection finder that matches your "2 µm" calibration architecture:
      - Use NI background acquisition timestamps (daq.time_of(i))
      - Use full Zaber position log + interpolation
      - Stop slewing immediately after the reflection interval ends

    Interval definition (hysteresis + debounce):
      start when v > threshold_high
      end when v <= threshold_low for debounce_s

    Returns either:
      mode="midpoint": event index = midpoint(idx_first, idx_last_above)
      mode="max":      event index = argmax(v) within the interval
    """
    if mode not in ("midpoint", "max"):
        raise ValueError(f"mode must be 'midpoint' or 'max', got {mode!r}")

    fs = float(getattr(ni, "sample_rate_hz", 1000.0))
    debounce_samples = max(1, int(round(float(debounce_s) * fs)))
    max_samples = int(fs * float(max_time_s)) + int(chunk_size) * 2  # a bit of headroom

    # Optional: flush FIFO / settle before starting the timed scan
    if pretrigger_flush:
        try:
            ni.flush()
        except Exception:
            pass

    # Interval state (indices are absolute indices into the acquisition stream == daq.values)
    in_interval = False
    idx_first: Optional[int] = None
    idx_last_above: Optional[int] = None
    low_run = 0

    # Track max within interval
    best_v = float("-inf")
    best_idx: Optional[int] = None

    last_idx = 0  # NI background write index cursor
    t_start = time.perf_counter()

    zlog = None
    daq = None

    try:
        # --- start DAQ acquisition (internal perf timeline model) ---
        ni.start_acquiring(max_samples=max_samples, chunk_size=int(chunk_size))

        # --- start Zaber position logging ---
        zaber.start_position_log(poll_s=float(z_poll_s) if z_poll_s > 0 else 0.0)

        # --- start motion ---
        zaber.start_slewing_guarded(float(speed_um_s), float(max_distance_um))

        while True:
            if (time.perf_counter() - t_start) > float(max_time_s):
                break

            rr, last_idx = ni.get_new_block(last_idx, copy=False)
            xs = rr.values
            n = int(xs.size)
            if n <= 0:
                time.sleep(0.001)
                continue

            # rr.t0_perf is the time of xs[0] (already aligned to acquisition t0)
            # Absolute index of xs[0] in the full acquisition stream:
            start_i = int(last_idx - n)

            # Scan this block sample-by-sample (1 kHz -> totally fine)
            th_hi = float(threshold_high)
            th_lo = float(threshold_low)

            for j in range(n):
                v = float(xs[j])
                i_abs = start_i + j  # absolute sample index (daq.values index)

                if not in_interval:
                    if v > th_hi:
                        in_interval = True
                        idx_first = i_abs
                        idx_last_above = i_abs
                        low_run = 0
                        best_v = v
                        best_idx = i_abs
                else:
                    # Update "last above high" and max tracking
                    if v > th_hi:
                        idx_last_above = i_abs
                        low_run = 0
                    # For max mode, keep tracking max anywhere inside the interval
                    if mode == "max" and v > best_v:
                        best_v = v
                        best_idx = i_abs

                    # End condition: baseline (low) for debounce_samples
                    if v <= th_lo:
                        low_run += 1
                        if low_run >= debounce_samples:
                            # Interval ended -> we can stop early
                            raise StopIteration
                    else:
                        low_run = 0

    except StopIteration:
        pass
    finally:
        # Stop motion/logging/acq (best effort)
        try:
            zaber.stop_slewing()
        except Exception:
            pass

        try:
            zlog = zaber.stop_position_log()
        except Exception:
            zlog = None

        try:
            daq = ni.stop_acquiring()
        except Exception:
            daq = None

    # Validate we detected something
    if (not in_interval) or idx_first is None or idx_last_above is None or daq is None:
        return ReflectionResult(found=False, mode=mode)

    # Choose event index
    if mode == "midpoint":
        event_i = int((idx_first + idx_last_above) // 2)
        peak_val = None
    else:
        if best_idx is None:
            event_i = int((idx_first + idx_last_above) // 2)
            peak_val = None
        else:
            event_i = int(best_idx)
            peak_val = float(best_v)

    # Use DAQ internal timestamp model (same as your calibration: daq.time_of(i))
    try:
        t_event = float(daq.time_of(event_i))
    except Exception:
        # Fallback if time_of isn't available for some reason
        t_event = float(daq.t0_perf) + (float(event_i) / float(daq.sample_rate_hz))

    # Interpolate Z from the full Zaber log (same monotonic cleanup as your calibration)
    if zlog is None or getattr(zlog, "t_perf", np.array([])).size < 2:
        try:
            z_event = float(zaber.get_position())
        except Exception:
            z_event = float("nan")
    else:
        z_event = _interp_z(t_event, np.asarray(zlog.t_perf), np.asarray(zlog.z_um))
    z_event = float(z_event) + float(z_offset_um)

    return ReflectionResult(
        found=True,
        mode=mode,
        event_index=event_i,
        event_time_perf=t_event,
        event_z_um=z_event,
        peak_value=peak_val,
        idx_first=int(idx_first),
        idx_last=int(idx_last_above),
    )


# --------------------
# Example usage
# --------------------
from brillouin_system.devices.ni.ni6008 import NI6008
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens

ni = NI6008()
z = ZaberEyeLens()

with ni.streaming():
    res = find_reflection_realtime(
        ni, z,
        mode="max",                  # or "midpoint"
        speed_um_s=5000.0,
        max_distance_um=5000.0,
        threshold_high=0.2,
        threshold_low=0.1,
        debounce_s=0.020,
        max_time_s=10.0,
        z_poll_s=0.016,
    )
print(res)
z.move_abs(res.event_z_um)