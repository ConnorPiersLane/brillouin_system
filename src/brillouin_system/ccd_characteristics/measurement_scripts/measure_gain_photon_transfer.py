"""Measure the camera sensitivity g [e-/count] by QUADRATIC photon transfer.

Result: sensitivity_e_per_count_preamp_1x in ccd_characteristics.toml.
Measured 2026-08-12 on 126 scans x 50 frames of the 2026-8-3 / 2026-8-5
water temperature series: g = 3.89 +- 0.04 e-/count (per day 3.91 / 3.90 —
hardware-stable), eps = 1.2-1.5% per day (session-dependent).

THE METHOD, and why it must be quadratic
----------------------------------------
Light arrives in Poisson-distributed lumps, so a pixel holding N electrons
has variance N. In counts alone that would give

    var[ADU^2] = S[ADU] / g            (g = electrons per count)

but on real spectra a second term is NOT negligible: the source carries a
per-frame common-mode intensity fluctuation (laser power / fiber coupling,
~1% rms, independently measured by the Stokes/anti-Stokes covariance
analysis), which every pixel sees multiplicatively. The honest model is

    var[ADU^2] = S/g + (eps*S)^2 + c

A LINEAR photon-transfer fit lumps the quadratic term into the slope and
returns a gain biased LOW by a session-dependent amount (2.9 on the August
series, 3.5 on July's — and that bias is why July's per-session estimates
spanned 2.7-4.8). The common mode scales the whole spectrum without moving
peak centres, so shot-noise predictions for the fitted SHIFT must use the
Poisson g. This script prints the linear-fit g alongside, so the bias is
visible every time.

TWO TRAPS, both of which we fell into first:
  * Slow drift. Over 50 frames the source drifts, inflating a naive
    temporal variance by ~22% at bright pixels. Take the variance from
    CONSECUTIVE-FRAME DIFFERENCES (var = var(diff)/2); anything slower than
    the frame rate cancels. (The ~1% common mode is frame-to-frame, so
    differencing does NOT remove it — hence the quadratic term.)
  * Outlier rejection on few frames. With k frames the per-pixel variance
    has k-1 degrees of freedom and a long right tail; rejecting "outliers"
    cuts that tail and biases the gain HIGH (measured: 3-frame chunks gave
    4.88 with rejection vs 3.32 without). Use >= 20 frames, do not reject.

RULES: never pool PTCs across sessions (eps is session-dependent; g is the
hardware number to compare across them). The definitive measurement — flat
uniform illumination at an exposure series, which removes the common-mode /
Poisson separation from the fit entirely — has still not been done
(~15 minutes at the instrument).

USAGE
    python measure_gain_photon_transfer.py scan1.pkl [scan2.pkl ...]

Each file: an AxialScan (or list of them) whose measurements are repeated
frames of a STATIC scene (fixed position, fixed illumination).
"""
from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np


def load_scans(path: str):
    if path.endswith((".h5", ".hdf5")):
        from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
        from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
            dict_to_dataclass_tree, load_dict_from_hdf5)
        loaded = dict_to_dataclass_tree(load_dict_from_hdf5(path), known_classes)
    else:
        with open(path, "rb") as f:
            loaded = pickle.load(f)
    return loaded if isinstance(loaded, list) else [loaded]


def photon_transfer(stack: np.ndarray, dark_level: float,
                    n_bins: int = 40) -> dict:
    """Quadratic PTC on one frame stack. Returns g, eps, c (+ linear g)."""
    n = stack.shape[0]
    if n < 20:
        print(f"  WARNING: only {n} frames — the recipe wants >= 20.")

    signal = stack.mean(axis=0) - dark_level                  # [counts]
    # Drift-immune per-pixel variance from consecutive differences.
    var = np.var(np.diff(stack, axis=0), axis=0, ddof=1) / 2.0

    s = signal.ravel()
    v = var.ravel()
    keep = np.isfinite(s) & np.isfinite(v) & (s > 0)
    s, v = s[keep], v[keep]

    # Bin by signal (median per bin) to stabilise the fit — NO rejection.
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    s_b, v_b = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum() >= 10:
            s_b.append(np.median(s[m]))
            v_b.append(np.median(v[m]))
    s_b, v_b = np.asarray(s_b), np.asarray(v_b)

    # var = c + S/g + (eps S)^2  — linear least squares in [1, S, S^2].
    a2, a1, a0 = np.polyfit(s_b, v_b, 2)
    lin_a1, _ = np.polyfit(s_b, v_b, 1)

    return {
        "g": 1.0 / a1 if a1 > 0 else float("nan"),
        "eps": float(np.sqrt(max(a2, 0.0))),
        "c": float(a0),
        "g_linear_biased": 1.0 / lin_a1 if lin_a1 > 0 else float("nan"),
        "n_frames": n,
        "max_signal": float(s_b.max()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="AxialScan .pkl/.h5 files "
                    "(repeated frames of a static scene)")
    args = ap.parse_args(argv)

    results = []
    for path in args.files:
        for scan in load_scans(path):
            stack = np.stack([np.asarray(m.frame_andor, dtype=float)
                              for m in scan.measurements])
            dark = getattr(scan.system_state, "dark_image", None)
            dark_level = (float(np.median(dark.mean_image))
                          if dark is not None
                          else float(np.median(stack)))
            r = photon_transfer(stack, dark_level)
            r["id"] = f"{path}:{scan.id}"
            results.append(r)
            print(f"{r['id']}: g = {r['g']:.3f} e-/count, "
                  f"eps = {100 * r['eps']:.2f}%, c = {r['c']:.2f}, "
                  f"(linear-PTC would give {r['g_linear_biased']:.3f} — "
                  f"biased low), n = {r['n_frames']} frames")

    gs = np.array([r["g"] for r in results if np.isfinite(r["g"])])
    if gs.size == 0:
        print("No usable results.")
        return 1
    print("\n==== Summary (one session at a time — never pool across sessions) ====")
    print(f"g = {np.median(gs):.3f} e-/count "
          f"(median of {gs.size} scans, spread {gs.std(ddof=1) if gs.size > 1 else 0:.3f})")
    print("\nPaste into ccd_characteristics.toml [ccd]:")
    print(f'sensitivity_e_per_count_preamp_1x = {np.median(gs):.2f}')
    print('sensitivity_measured = "<date>"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
