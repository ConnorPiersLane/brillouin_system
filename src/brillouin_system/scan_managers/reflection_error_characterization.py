"""
Reflection-plane finding: error characterization (acquisition script).

Purpose
-------
The PI wants an estimate of the measurement error of the realtime reflection
plane finder (ni_reflection_finder4.find_reflection_realtime). This script
measures both the *true* location of the reflection plane and the scatter of
the realtime finder, so a second script (reflection_error_analysis.py) can
separate:

  - bias      : mean(finder estimates) - static ("true") plane center
  - precision : std(finder estimates)

Protocol (matches the plan agreed with the PI)
----------------------------------------------
1. Rough find:
     Run the realtime finder once with the current GUI settings
     (scanning_config.toml, section [axial_scanning]) to locate the plane
     approximately. The lens must be positioned BEHIND the plane (same
     starting condition as the GUI workflow) before starting this script.

2. Static slow scan (repeated N_SLOW_SCANS times, default 3):
     Move SLOW_SCAN_BEHIND_UM (default 100 um) behind the rough plane, then
     step the lens forward in SLOW_SCAN_STEP_UM (default 2 um) increments up
     to SLOW_SCAN_AHEAD_UM past the rough plane. At every step the stage is
     stationary: settle, flush the NI FIFO (discards samples acquired during
     the move), then read a block of DAQ samples. This yields the static
     reflection profile signal-vs-z, free of any motion/timing errors.
     Backlash control: the scan start is always approached from further
     behind, and steps are monotonic forward.

3. Finder trials (N_FINDER_TRIALS times, default 50):
     Move to a fixed start position TRIAL_APPROACH_UM behind the rough plane
     (with backlash preload) and run find_reflection_realtime with the
     current GUI settings. Store where the finder claims the plane is
     (event_z_um, WITHOUT the constant z_offset_um the GUI adds afterwards).

4. Everything is written to a human-readable, timestamped JSON file in
   OUTPUT_DIR (created next to this script).

Analysis / plots: see reflection_error_analysis.py in this folder.

Usage
-----
Position the lens behind the reflection plane (as for a normal GUI reflection
search), then run:

    python -m brillouin_system.scan_managers.reflection_error_characterization

Tunables are the module-level constants below. TRIAL_DIRECTION defaults to
"alternate" (forward/backward interleaved): timing latency between the DAQ
and the Zaber position log shifts the estimate along the travel direction,
so averaging forward and backward means cancels it and yields an accurate,
latency-corrected plane position; half the forward-backward difference
quantifies the latency bias itself.
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from brillouin_system.devices.ni.ni6008 import NI6008
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_position_log import interp_z_positions
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult, find_reflection_realtime
from brillouin_system.scan_managers.scanning_config.scanning_config import (
    AXIAL_SCANNING_TOML_PATH,
    ScanningConfig,
    load_axial_scanning_config,
)

# ----------------------------
# Protocol constants
# ----------------------------
SLOW_SCAN_BEHIND_UM = 100.0    # scan starts this far behind the rough plane
SLOW_SCAN_AHEAD_UM = 100.0     # ... and ends this far past it (captures the full peak)
SLOW_SCAN_STEP_UM = 2.0        # step size of the static scan
N_SLOW_SCANS = 3               # number of static scan repeats
STEP_SETTLE_S = 0.05           # wait after each step before reading the DAQ
STEP_READ_S = 0.05             # DAQ read duration per step (n = fs * STEP_READ_S)
BACKLASH_PRELOAD_UM = 100.0    # approach distance so moves always end going forward

N_FINDER_TRIALS = 50           # number of realtime finder repeats
TRIAL_APPROACH_UM = 1000.0     # finder trials start this far behind/in front of the rough plane
# "alternate": interleave forward/backward runs. Any DAQ<->Zaber timing latency
# shifts the estimate ALONG the travel direction, so
#   (mean_fwd + mean_bwd) / 2  ->  latency-corrected plane position
#   (mean_fwd - mean_bwd) / 2  ->  latency-induced bias per direction
# Set to "forward" to only replicate the GUI's forward search.
TRIAL_DIRECTION = "alternate"  # "alternate" | "forward"

OUTPUT_DIR = Path(__file__).parent / "reflection_error_data"


# ----------------------------
# Helpers
# ----------------------------

def _move_with_backlash_preload(zaber: ZaberEyeLens, target_um: float) -> None:
    """Move to target_um such that the final approach is always forward (+z)."""
    zaber.move_abs(target_um - BACKLASH_PRELOAD_UM)
    zaber.move_abs(target_um)


def _reflection_result_to_dict(res: ReflectionResult) -> dict:
    """Summarize a ReflectionResult for JSON (no raw traces; those are large)."""
    d = {
        "found": bool(res.found),
        "event_z_um": _f(res.event_z_um),
        "peak_value": _f(res.peak_value),
        "background_mean": _f(res.background_mean),
        "background_std": _f(res.background_std),
        "threshold_high": _f(res.threshold_high),
        "threshold_low": _f(res.threshold_low),
        "z_first_um": None,
        "z_last_um": None,
    }
    # Map the first/last above-threshold sample to z via the Zaber log:
    # gives the detected interval as a z-range, useful for width diagnostics.
    if (
        res.found
        and res.idx_first is not None
        and res.idx_last is not None
        and res.daq_ts is not None
        and res.zaber_lens_ts is not None
        and res.zaber_lens_ts.size >= 2
    ):
        t_first = float(res.daq_ts[res.idx_first])
        t_last = float(res.daq_ts[res.idx_last])
        zt = np.asarray(res.zaber_lens_ts)
        zz = np.asarray(res.zaber_lens_z_um)
        d["z_first_um"] = _f(interp_z_positions(t_first, zt, zz))
        d["z_last_um"] = _f(interp_z_positions(t_last, zt, zz))
    return d


def _f(x) -> float | None:
    """float() or None, JSON-safe."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[save] wrote {path}")


# ----------------------------
# Protocol phases
# ----------------------------

def run_rough_find(ni: NI6008, zaber: ZaberEyeLens, cfg: ScanningConfig) -> ReflectionResult:
    """Phase 1: one realtime finder run with the current GUI settings."""
    print("[rough] running realtime finder to locate the plane approximately ...")
    res = find_reflection_realtime(
        ni,
        zaber,
        ni_sample_rate_hz=cfg.ni_sample_rate_hz,
        speed_um_s=cfg.speed_um_s,
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
    )
    if not res.found or res.event_z_um is None or not np.isfinite(res.event_z_um):
        raise RuntimeError(
            "Rough reflection find failed. Make sure the lens starts behind the "
            "reflection plane and the current scanning_config.toml settings work in the GUI."
        )
    print(f"[rough] plane approximately at z = {res.event_z_um:.2f} um")
    return res


def run_slow_scan(
    ni: NI6008,
    zaber: ZaberEyeLens,
    cfg: ScanningConfig,
    rough_z_um: float,
    scan_index: int,
) -> dict:
    """
    Phase 2: one static scan across the plane.

    Steps SLOW_SCAN_STEP_UM forward from (rough - SLOW_SCAN_BEHIND_UM) to
    (rough + SLOW_SCAN_AHEAD_UM); at each stationary step reads a DAQ block.
    """
    z_start = rough_z_um - SLOW_SCAN_BEHIND_UM
    z_end = rough_z_um + SLOW_SCAN_AHEAD_UM
    z_targets = np.arange(z_start, z_end + 0.5 * SLOW_SCAN_STEP_UM, SLOW_SCAN_STEP_UM)

    ni.set_sample_rate_hz(cfg.ni_sample_rate_hz)
    fs = ni.get_sample_rate_hz()
    n_per_step = max(1, int(round(STEP_READ_S * fs)))

    print(f"[slow {scan_index + 1}/{N_SLOW_SCANS}] {z_targets.size} steps, "
          f"{z_start:.1f} -> {z_end:.1f} um, {n_per_step} samples/step")

    _move_with_backlash_preload(zaber, float(z_targets[0]))

    steps = []
    with ni.streaming():
        for k, z_t in enumerate(z_targets):
            zaber.move_abs(float(z_t))
            time.sleep(STEP_SETTLE_S)
            z_actual = float(zaber.get_position())

            ni.flush()  # discard samples acquired during the move/settle
            values = np.asarray(ni.read_block(n_per_step, timeout_s=STEP_READ_S + 1.0))

            steps.append({
                "z_target_um": float(z_t),
                "z_actual_um": z_actual,
                "daq_mean": float(np.mean(values)),
                "daq_std": float(np.std(values)),
                "n_samples": int(values.size),
                "daq_values": [float(v) for v in values],
            })
            if (k + 1) % 20 == 0:
                print(f"[slow {scan_index + 1}] step {k + 1}/{z_targets.size}")

    return {
        "scan_index": scan_index,
        "step_um": SLOW_SCAN_STEP_UM,
        "settle_s": STEP_SETTLE_S,
        "read_s": STEP_READ_S,
        "sample_rate_hz": float(fs),
        "steps": steps,
    }


def run_finder_trials(
    ni: NI6008,
    zaber: ZaberEyeLens,
    cfg: ScanningConfig,
    rough_z_um: float,
) -> list[dict]:
    """
    Phase 3: repeated realtime finder runs from a fixed start position.

    Forward trials start TRIAL_APPROACH_UM behind the plane; backward trials
    (TRIAL_DIRECTION="alternate") start the same distance in front of it.
    """
    trials = []
    for i in range(N_FINDER_TRIALS):
        forwards = True
        if TRIAL_DIRECTION == "alternate" and (i % 2 == 1):
            forwards = False

        if forwards:
            start_z = rough_z_um - TRIAL_APPROACH_UM
            speed = abs(cfg.speed_um_s)
        else:
            start_z = rough_z_um + TRIAL_APPROACH_UM
            speed = -abs(cfg.speed_um_s)

        _move_with_backlash_preload(zaber, start_z)

        res = find_reflection_realtime(
            ni,
            zaber,
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
        )

        d = _reflection_result_to_dict(res)
        d["trial_index"] = i
        d["direction"] = "forward" if forwards else "backward"
        d["start_z_um"] = float(start_z)
        trials.append(d)

        z_str = f"{d['event_z_um']:.2f}" if d["event_z_um"] is not None else "---"
        print(f"[trial {i + 1}/{N_FINDER_TRIALS}] {d['direction']:8s} "
              f"found={d['found']} event_z={z_str} um")
    return trials


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    cfg = load_axial_scanning_config(AXIAL_SCANNING_TOML_PATH, "axial_scanning")
    print("[config] using current GUI settings from scanning_config.toml:")
    for k, v in asdict(cfg).items():
        print(f"    {k} = {v}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"reflection_error_{stamp}.json"

    data: dict = {
        "meta": {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "script": Path(__file__).name,
            "config": asdict(cfg),
            "protocol": {
                "slow_scan_behind_um": SLOW_SCAN_BEHIND_UM,
                "slow_scan_ahead_um": SLOW_SCAN_AHEAD_UM,
                "slow_scan_step_um": SLOW_SCAN_STEP_UM,
                "n_slow_scans": N_SLOW_SCANS,
                "step_settle_s": STEP_SETTLE_S,
                "step_read_s": STEP_READ_S,
                "backlash_preload_um": BACKLASH_PRELOAD_UM,
                "n_finder_trials": N_FINDER_TRIALS,
                "trial_approach_um": TRIAL_APPROACH_UM,
                "trial_direction": TRIAL_DIRECTION,
            },
            "note": (
                "event_z_um values are RAW finder estimates, without the constant "
                "z_offset_um the GUI adds afterwards."
            ),
        },
        "rough_find": None,
        "slow_scans": [],
        "finder_trials": [],
    }

    ni = NI6008()
    zaber = ZaberEyeLens(home_on_connect=False)  # attach without moving the lens

    try:
        # Phase 1: rough localization
        rough = run_rough_find(ni, zaber, cfg)
        rough_z = float(rough.event_z_um)
        data["rough_find"] = _reflection_result_to_dict(rough)

        # Phase 2: static slow scans
        for s in range(N_SLOW_SCANS):
            data["slow_scans"].append(run_slow_scan(ni, zaber, cfg, rough_z, s))

        # Phase 3: realtime finder trials
        data["finder_trials"] = run_finder_trials(ni, zaber, cfg, rough_z)

    finally:
        # Save whatever was collected, even on abort/error.
        _save_json(data, out_path)
        try:
            zaber.close()
        except Exception:
            pass

    n_found = sum(1 for t in data["finder_trials"] if t["found"])
    print(f"\nDone. {len(data['slow_scans'])} slow scans, "
          f"{n_found}/{len(data['finder_trials'])} finder trials found the plane.")
    print(f"Next: python -m brillouin_system.scan_managers.reflection_error_analysis \"{out_path}\"")


if __name__ == "__main__":
    main()
