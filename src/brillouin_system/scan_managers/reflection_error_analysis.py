"""
Reflection-plane finding: error analysis (companion to
reflection_error_characterization.py).

Answers the question: where is the REAL reflection plane, and what is the
measurement error of the realtime finder?

Input
-----
A JSON file produced by reflection_error_characterization.py. Pass the path
as the first command-line argument, or run without arguments to analyze the
newest file in scan_managers/reflection_error_data/.

    python -m brillouin_system.scan_managers.reflection_error_analysis [file.json]

What it computes
----------------
1. "True" plane from the static slow scans (stage stationary at every point,
   so no motion/timing errors). Per scan:
     - peak z            : z of the maximum DAQ signal
     - half-max center   : midpoint between the rising and falling half-max
                           crossings (robust against an asymmetric peak top)
     - centroid          : signal-weighted centroid above the half-max level
     - FWHM              : physical width of the reflection profile
                           (spot size / interface response - NOT finder error)
   The true plane center is the mean half-max center across scans; its std
   across scans shows the repeatability of the static measurement itself.

2. Finder statistics from the repeated realtime trials:
     - mean, std, min/max, peak-to-peak of event_z_um (per direction if the
       acquisition used alternating directions)

3. Error decomposition:
     - bias      = mean(finder) - true center   (systematic offset, e.g. from
                   DAQ<->Zaber timing latency during motion)
     - precision = std(finder)                  (trial-to-trial scatter)

4. If both directions are present (TRIAL_DIRECTION="alternate"):
     - corrected center = (mean_fwd + mean_bwd) / 2
         Latency shifts the estimate along the travel direction, so this
         average cancels the latency bias -> accurate plane position from
         the realtime finder alone.
     - latency bias     = (mean_fwd - mean_bwd) / 2
         The systematic latency-induced offset of a single-direction search.
     The corrected center is compared against the static true center: their
     residual difference is the direction-independent offset that alternation
     cannot remove.

Output: printed report + a two-panel figure (static profiles with the true
plane region, and a histogram of the finder estimates), saved as PNG next to
the JSON file and shown interactively.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent / "reflection_error_data"


# ----------------------------
# Slow-scan profile analysis
# ----------------------------

def _cross(z0: float, v0: float, z1: float, v1: float, level: float) -> float:
    """Linear interpolation of the z where the signal crosses `level`."""
    if v1 == v0:
        return z0
    return z0 + (level - v0) / (v1 - v0) * (z1 - z0)


def analyze_slow_scan(scan: dict) -> dict:
    """Estimate plane center and width from one static scan."""
    steps = scan["steps"]
    z = np.array([s["z_actual_um"] for s in steps], dtype=float)
    v = np.array([s["daq_mean"] for s in steps], dtype=float)

    # Background from the first few points (scan starts well behind the plane)
    n_bg = max(3, int(0.05 * v.size))
    bg = float(np.median(v[:n_bg]))

    i_peak = int(np.argmax(v))
    peak = float(v[i_peak])
    half = bg + 0.5 * (peak - bg)

    # Rising half-max crossing (walk left from the peak)
    i = i_peak
    while i > 0 and v[i - 1] >= half:
        i -= 1
    z_rise = _cross(z[i - 1], v[i - 1], z[i], v[i], half) if i > 0 else float(z[0])

    # Falling half-max crossing (walk right from the peak)
    j = i_peak
    while j < v.size - 1 and v[j + 1] >= half:
        j += 1
    z_fall = _cross(z[j], v[j], z[j + 1], v[j + 1], half) if j < v.size - 1 else float(z[-1])

    # Signal-weighted centroid above the half-max level
    w = np.clip(v - half, 0.0, None)
    centroid = float(np.sum(w * z) / np.sum(w)) if np.sum(w) > 0 else float(z[i_peak])

    return {
        "z": z,
        "v": v,
        "background": bg,
        "peak_value": peak,
        "z_peak": float(z[i_peak]),
        "z_rise": float(z_rise),
        "z_fall": float(z_fall),
        "center_halfmax": 0.5 * (float(z_rise) + float(z_fall)),
        "centroid": centroid,
        "fwhm": float(z_fall) - float(z_rise),
    }


# ----------------------------
# Report + plot
# ----------------------------

def main() -> None:
    # --- locate input file ---
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        candidates = sorted(DATA_DIR.glob("reflection_error_*.json"))
        if not candidates:
            raise SystemExit(f"No reflection_error_*.json files found in {DATA_DIR}")
        path = candidates[-1]
    print(f"[load] {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # --- 1) true plane from the static scans ---
    scan_results = [analyze_slow_scan(s) for s in data["slow_scans"]]
    if not scan_results:
        raise SystemExit("No slow scans in file - nothing to compare against.")

    centers = np.array([r["center_halfmax"] for r in scan_results])
    fwhms = np.array([r["fwhm"] for r in scan_results])
    true_center = float(np.mean(centers))
    true_center_std = float(np.std(centers))

    print("\n=== Static slow scans (true plane) ===")
    for k, r in enumerate(scan_results):
        print(f"  scan {k}: half-max center = {r['center_halfmax']:9.2f} um   "
              f"peak z = {r['z_peak']:9.2f} um   centroid = {r['centroid']:9.2f} um   "
              f"FWHM = {r['fwhm']:6.2f} um")
    print(f"  -> TRUE plane center = {true_center:.2f} um "
          f"(std across {centers.size} scans: {true_center_std:.2f} um)")
    print(f"  -> mean FWHM (optical width, spot size etc.): {np.mean(fwhms):.2f} um")

    # --- 2) finder trial statistics ---
    trials = data.get("finder_trials", [])
    ok = [t for t in trials if t.get("found") and t.get("event_z_um") is not None]
    n_missed = len(trials) - len(ok)
    if not ok:
        raise SystemExit("No successful finder trials in file.")

    est = np.array([t["event_z_um"] for t in ok], dtype=float)
    directions = np.array([t.get("direction", "forward") for t in ok])

    print("\n=== Realtime finder trials ===")
    print(f"  trials: {len(trials)} total, {len(ok)} found, {n_missed} missed")

    def _stats(x: np.ndarray, label: str) -> None:
        print(f"  {label:9s}: n = {x.size:3d}   mean = {np.mean(x):9.2f} um   "
              f"std = {np.std(x, ddof=1) if x.size > 1 else 0.0:6.2f} um   "
              f"min..max = {np.min(x):.2f} .. {np.max(x):.2f} um "
              f"(p-p {np.ptp(x):.2f} um)")

    _stats(est, "all")
    for d in np.unique(directions):
        if np.unique(directions).size > 1:
            _stats(est[directions == d], d)

    # --- 3) error decomposition ---
    bias = float(np.mean(est)) - true_center
    precision = float(np.std(est, ddof=1)) if est.size > 1 else 0.0
    print("\n=== Error of the reflection plane finding ===")
    print(f"  bias      (mean finder - true center): {bias:+.2f} um")
    print(f"  precision (std of finder estimates)  : {precision:.2f} um")
    print(f"  (true-center repeatability            : {true_center_std:.2f} um)")

    # --- 4) forward/backward decomposition (alternate mode) ---
    corrected_center = None
    est_fwd = est[directions == "forward"]
    est_bwd = est[directions == "backward"]
    if est_fwd.size > 0 and est_bwd.size > 0:
        mean_fwd = float(np.mean(est_fwd))
        mean_bwd = float(np.mean(est_bwd))
        corrected_center = 0.5 * (mean_fwd + mean_bwd)
        latency_bias = 0.5 * (mean_fwd - mean_bwd)
        # Standard error of the corrected center from the two direction means
        sem = 0.5 * np.hypot(
            np.std(est_fwd, ddof=1) / np.sqrt(est_fwd.size) if est_fwd.size > 1 else 0.0,
            np.std(est_bwd, ddof=1) / np.sqrt(est_bwd.size) if est_bwd.size > 1 else 0.0,
        )
        print("\n=== Direction decomposition (latency-corrected estimate) ===")
        print(f"  mean forward                : {mean_fwd:9.2f} um")
        print(f"  mean backward               : {mean_bwd:9.2f} um")
        print(f"  corrected center (fwd+bwd)/2: {corrected_center:9.2f} +- {sem:.2f} um")
        print(f"  latency bias (fwd-bwd)/2    : {latency_bias:+9.2f} um per direction")
        print(f"  corrected - static true     : {corrected_center - true_center:+9.2f} um "
              f"(direction-independent residual)")

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for k, r in enumerate(scan_results):
        ax1.plot(r["z"], r["v"], marker=".", lw=1, label=f"slow scan {k}")
    r0 = scan_results[0]
    ax1.axvspan(min(r["z_rise"] for r in scan_results),
                max(r["z_fall"] for r in scan_results),
                alpha=0.12, color="tab:green", label="FWHM region (static)")
    ax1.axvline(true_center, color="tab:green", ls="--", lw=1.5,
                label=f"true center {true_center:.1f} um")
    ax1.axhline(r0["background"], color="gray", ls=":", lw=1, label="background")
    ax1.set_ylabel("DAQ signal (V)")
    ax1.set_title(f"Reflection plane: static profile vs. realtime finder\n{path.name}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    bin_width = max(0.5, np.ptp(est) / 15) if est.size > 1 else 1.0
    bins = np.arange(est.min() - bin_width, est.max() + 2 * bin_width, bin_width)
    if np.unique(directions).size > 1:
        for d, color in zip(np.unique(directions), ["tab:blue", "tab:orange"]):
            ax2.hist(est[directions == d], bins=bins, alpha=0.6, color=color,
                     label=f"finder ({d}, n={np.sum(directions == d)})")
    else:
        ax2.hist(est, bins=bins, alpha=0.7, color="tab:blue",
                 label=f"finder estimates (n={est.size})")
    ax2.axvline(true_center, color="tab:green", ls="--", lw=1.5, label="true center")
    ax2.axvline(float(np.mean(est)), color="tab:red", ls="-", lw=1.5,
                label=f"finder mean (bias {bias:+.1f} um)")
    if corrected_center is not None:
        ax2.axvline(corrected_center, color="tab:purple", ls="-.", lw=1.5,
                    label=f"corrected (fwd+bwd)/2 = {corrected_center:.1f} um")
    ax2.set_xlabel("lens z (um)")
    ax2.set_ylabel("count")
    ax2.legend(fontsize=8,
               title=f"bias {bias:+.2f} um, precision (std) {precision:.2f} um")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    print(f"\n[plot] saved {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
