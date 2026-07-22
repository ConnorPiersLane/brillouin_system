from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_position_log import interp_z_positions, ZaberPositionLog


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    """
    Result of a reflection search.

    Indices (event_index / idx_first / idx_last) are acquisition-buffer indices:
      - i == 0 refers to the first sample stored by NI6008 background acquisition
      - the corresponding time is ni_result.time_of(i) (perf_counter timeline model)
      - event_index is FRACTIONAL (centroid estimator, sub-sample resolution)
    """
    found: bool
    event_index: Optional[float] = None        # fractional acquisition-buffer index (centroid)
    event_time_perf: Optional[float] = None    # perf_counter time_of(event_index)
    event_z_um: Optional[float] = None         # z at event time (line fit; falls back to interp)
    event_z_um_interp: Optional[float] = None  # z via 2-point interpolation of the Zaber log
    event_z_um_fit: Optional[float] = None     # z via constant-velocity line fit (NaN-> None if fit failed)
    z_offset_um: Optional[float] = None
    peak_value: Optional[float] = None
    background_mean: Optional[float] = None
    background_std: Optional[float] = None
    threshold_high: Optional[float] = None
    threshold_low: Optional[float] = None
    idx_first: Optional[int] = None            # first sample above threshold_high
    idx_last: Optional[int] = None             # last sample above threshold_high (within the interval)
    n_samples_above: Optional[int] = None      # samples above threshold_high in the accepted interval
    n_rejected_intervals: int = 0              # intervals rejected as noise spikes (< min_samples_above)
    daq_ts: Optional[np.ndarray] = None
    daq_values: Optional[np.ndarray] = None
    zaber_lens_ts: Optional[np.ndarray] = None
    zaber_lens_z_um: Optional[np.ndarray] = None

def print_reflection_result(result):
    for f in fields(result):
        value = getattr(result, f.name)
        print(f"{f.name}: {value}")


def fit_z_at_time(
    t_query: float,
    t_z: np.ndarray,
    z_um: np.ndarray,
    *,
    slope_tol: float = 0.2,
    min_points: int = 8,
) -> float:
    """
    Least-squares line fit z(t) over the constant-velocity portion of the
    Zaber position log, evaluated at t_query.

    During the search the stage slews at constant velocity, so z(t) is a
    straight line; fitting all log samples averages out the per-sample
    timestamp jitter (~ms serial latency placement), which 2-point
    interpolation inherits in full. The acceleration ramp at the start and
    the deceleration after the early stop are excluded by keeping only
    segments whose local slope is within slope_tol of the median slope.

    Returns NaN if fewer than min_points samples remain — caller should fall
    back to interp_z_positions().
    """
    t = np.asarray(t_z, dtype=np.float64)
    z = np.asarray(z_um, dtype=np.float64)
    if t.size < min_points:
        return float("nan")

    keep = np.concatenate(([True], np.diff(t) > 0))
    t = t[keep]
    z = z[keep]
    if t.size < min_points:
        return float("nan")

    v = np.diff(z) / np.diff(t)
    v_ref = float(np.median(v))
    if v_ref == 0.0 or not np.isfinite(v_ref):
        return float("nan")

    good_seg = np.abs(v - v_ref) <= slope_tol * abs(v_ref)
    # a sample belongs to the constant-velocity window if an adjacent segment is good
    good_pt = np.zeros(t.size, dtype=bool)
    good_pt[:-1] |= good_seg
    good_pt[1:] |= good_seg
    if int(np.sum(good_pt)) < min_points:
        return float("nan")

    # center t for numerical conditioning (perf_counter values are large)
    t0 = float(np.mean(t[good_pt]))
    coef = np.polyfit(t[good_pt] - t0, z[good_pt], 1)
    return float(np.polyval(coef, float(t_query) - t0))


def find_reflection_realtime(
    ni: NI6008,
    zaber: ZaberEyeLens,
    *,
    ni_sample_rate_hz: float,
    speed_um_s: float,
    max_distance_um: float,
    threshold_high_n_sigma: int,
    threshold_low_n_sigma: int,
    bg_acqui_s: float,
    debounce_s: float = 0.020,     # 20 ms at 1 kHz is a good start
    z_poll_s: float = 0.016,       #min 16ms - max ~63 Hz;"
    alpha: float = 0.25,
    chunk_size: int = 1024,
    idle_sleep_s: float = 0.001,   # sleep when DAQ has no new samples
    z_offset_um: float = 0.0,      # optional manual offset
    centroid_fraction: float = 0.8,  # centroid weights: samples above this fraction of peak amplitude
    min_samples_above: int = 3,    # interval acceptance: reject noise spikes with fewer samples above threshold_high
) -> ReflectionResult:
    """
    Early-stop reflection finder using:
      - NI6008 background acquisition indices/timestamps (perf_counter model)
      - Zaber position log + interpolation
      - guarded slewing stop immediately after reflection interval ends

    Interval definition (hysteresis + debounce):
      - start when v > threshold_high
      - end when v <= threshold_low for debounce_s continuously

    Interval acceptance (false-trigger rejection, power-independent):
      An ended interval is only accepted as the reflection if it contains at
      least min_samples_above samples above threshold_high. Single-sample
      noise spikes (observed when the background window catches a quiet noise
      state and the n-sigma threshold lands near the spike amplitude) are
      rejected and the search continues slewing; rejected intervals are
      counted in n_rejected_intervals.

    Event choice (centroid estimator, sub-sample resolution):
      event_index = signal-weighted center of mass of the samples between
      idx_first and idx_last, using weights max(0, v - level) with
      level = background + centroid_fraction * (peak - background).
      This averages the noise of all samples near the peak top (argmax jitters
      across a flat/noisy plateau and is undefined on a clipped peak) and
      yields a FRACTIONAL buffer index, i.e. finer than the 1-sample z spacing
      (speed_um_s / ni_sample_rate_hz). peak_value still reports the argmax value.
    """

    ni.set_sample_rate_hz(ni_sample_rate_hz)
    fs = ni.get_sample_rate_hz()

    with ni.streaming():
        ni.flush()
        bg = ni.read_block(int(bg_acqui_s*fs), timeout_s=bg_acqui_s+1)
        ni.flush()

        av = np.mean(bg)
        sigma = max(np.std(bg), 0.001)
        th_hi = av + threshold_high_n_sigma * sigma
        th_lo = av + threshold_low_n_sigma * sigma

        max_time_s = abs(max_distance_um / speed_um_s) + 1
        debounce_samples = max(1, int(round(float(debounce_s) * fs)))

        # Interval state (indices are acquisition-buffer indices)
        in_interval = False
        idx_first: Optional[int] = None
        idx_last_above: Optional[int] = None
        low_run = 0
        n_above = 0
        n_rejected = 0
        daq: NIReadResult | None = None
        zlog: ZaberPositionLog | None = None

        # Track max within interval (for mode="max")
        best_v = float("-inf")
        best_idx: Optional[int] = None
        rr: NIReadResult | None = None

        # Cursor into NI acquisition-buffer indices for incremental reads
        last_idx = 0

        t_start = time.perf_counter()

        try:
            # --- start DAQ acquisition ---
            ni.start_acquiring(
                max_sampling_time_s=max_time_s,
                chunk_size=chunk_size,
                idle_sleep_s=idle_sleep_s,
            )

            # --- start Zaber position logging ---
            zaber.start_position_log(poll_s=z_poll_s, alpha=alpha)

            # --- start motion ---
            zaber.start_slewing_guarded(speed_um_s, max_distance_um)


            while True:
                if (time.perf_counter() - t_start) > max_time_s:
                    break

                rr: NIReadResult = ni.get_new_block_result(last_idx, copy=False)
                xs = rr.values
                n = xs.size
                if n <= 0:
                    time.sleep(idle_sleep_s)
                    continue

                # IMPORTANT: rr.ind0 is the acquisition-buffer index of xs[0]
                start_i = rr.ind0
                last_idx = start_i + n  # advance cursor for next call

                # Scan this block sample-by-sample (1 kHz -> fine)
                for j in range(n):
                    v = xs[j]
                    i_abs = start_i + j

                    if not in_interval:
                        if v > th_hi:
                            in_interval = True
                            idx_first = i_abs
                            idx_last_above = i_abs
                            low_run = 0
                            n_above = 1
                            best_v = v
                            best_idx = i_abs
                    else:
                        # last above high
                        if v > th_hi:
                            idx_last_above = i_abs
                            low_run = 0
                            n_above += 1

                        # track max anywhere in interval if requested
                        if v > best_v:
                            best_v = v
                            best_idx = i_abs

                        # end condition: below low for debounce_samples
                        if v <= th_lo:
                            low_run += 1
                            if low_run >= debounce_samples:
                                if n_above >= min_samples_above:
                                    raise StopIteration
                                # too few samples above threshold: noise spike,
                                # reject interval and keep searching
                                n_rejected += 1
                                in_interval = False
                                idx_first = None
                                idx_last_above = None
                                low_run = 0
                                n_above = 0
                                best_v = float("-inf")
                                best_idx = None
                        else:
                            low_run = 0

        except StopIteration:
            pass
        finally:
            try: zaber.stop_slewing()
            except Exception: pass
            try:
                zlog = zaber.stop_position_log()
            except Exception:
                zlog = None

            try: err = ni.get_acquiring_error()
            except Exception: err = None

            try:
                daq: NIReadResult = ni.stop_acquiring()
            except Exception:
                daq: None = None


    if (
        (not in_interval)
        or idx_first is None
        or idx_last_above is None
        or daq is None
        or best_idx is None
        or n_above < min_samples_above  # timeout inside a spurious interval
    ):
        # Validate detection + acquisition
        return ReflectionResult(
            found=False,
            n_rejected_intervals=n_rejected,
        )

    # Interpolate Z from full Zaber log
    if zlog is None or zlog.t_perf.size < 2:
        return ReflectionResult(found=False, n_rejected_intervals=n_rejected)

    daq_ts = np.asarray(daq.timestamps_perf())
    daq_values = np.asarray(daq.values)
    zaber_lens_ts = np.asarray(zlog.t_perf)
    zaber_lens_z_um = np.asarray(zlog.z_um)

    # Centroid event estimator: center of mass of the samples near the peak
    # top -> fractional buffer index (sub-sample resolution, robust against
    # plateau noise and clipping; see docstring).
    seg = daq_values[int(idx_first): int(idx_last_above) + 1]
    level = av + centroid_fraction * (best_v - av)
    w = np.clip(seg - level, 0.0, None)
    w_sum = float(np.sum(w))
    if w_sum > 0.0:
        event_i = float(idx_first) + float(np.sum(w * np.arange(seg.size))) / w_sum
    else:
        event_i = float(best_idx)  # degenerate interval: fall back to argmax
    peak_val = best_v
    t_event = daq.time_of(event_i)

    # z at event time: constant-velocity line fit over the whole Zaber log
    # (averages per-sample timestamp jitter), 2-point interpolation as
    # fallback and for comparison.
    z_interp = interp_z_positions(t_event, zaber_lens_ts, zaber_lens_z_um)
    z_fit = fit_z_at_time(t_event, zaber_lens_ts, zaber_lens_z_um)
    z_event = z_fit if np.isfinite(z_fit) else z_interp




    return ReflectionResult(
        found=True,
        event_index=event_i,
        event_time_perf=t_event,
        event_z_um=z_event,
        event_z_um_interp=float(z_interp) if np.isfinite(z_interp) else None,
        event_z_um_fit=float(z_fit) if np.isfinite(z_fit) else None,
        z_offset_um=z_offset_um,
        peak_value=peak_val,
        background_mean=av,
        background_std=sigma,
        threshold_high=th_hi,
        threshold_low=th_lo,
        idx_first=int(idx_first),
        idx_last=int(idx_last_above),
        n_samples_above=int(n_above),
        n_rejected_intervals=int(n_rejected),
        daq_ts = daq_ts,
        daq_values = daq_values,
        zaber_lens_ts = zaber_lens_ts,
        zaber_lens_z_um = zaber_lens_z_um,
    )


# --------------------
# Example usage
# --------------------
if __name__ == "__main__":
    from brillouin_system.devices.ni.ni6008 import NI6008, NIReadResult
    from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens

    ni = NI6008()
    z = ZaberEyeLens()
    z.move_abs(6000)
    res = find_reflection_realtime(
        ni, z,
        ni_sample_rate_hz=1000,
        speed_um_s=2000.0,
        max_distance_um=10000.0,
        threshold_high_n_sigma=20,
        threshold_low_n_sigma=4,
        bg_acqui_s=0.1,
        debounce_s=0.020,
        z_poll_s=0.016,
        chunk_size=2048,
        idle_sleep_s=0.001,
    )

    # print_reflection_result(res)

    if res.found and res.event_z_um is not None and np.isfinite(res.event_z_um):
        z.move_abs(res.event_z_um)
        print(res.event_z_um)
    # plt.figure()
    # # plt.plot(r[:500], marker='o')
    # plt.plot(res.zaber_lens_ts, res.zaber_lens_z_um, marker='o')
    # plt.axvline(x=res.event_time_perf)  # vertical line at xe
    #
    # plt.grid(True)
    #
    # plt.show()