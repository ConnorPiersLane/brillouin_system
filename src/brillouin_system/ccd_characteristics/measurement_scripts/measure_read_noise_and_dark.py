"""Measure read noise and the dark/bias level from closed-shutter frames.

Results: read_noise_counts and dark_median_counts in ccd_characteristics
.toml. Measured 2026-08-19 on a closed-shutter dark exposure ladder:
read noise 1.10 counts rms (= 4.3 e- at g = 3.89), dark median ~200
counts/px, and NO measurable dark current at operating settings (-20 C,
exposures up to ~1 s).

THE METHOD
----------
* Read noise: a closed-shutter frame's per-pixel temporal std IS the read
  noise, because there is no dark current at our settings — verified by
  the exposure ladder: the std does not grow with exposure. It lives in
  the READOUT amplifier, so it is exposure-independent but DOES depend on
  the readout mode (amplifier / speed / preamp) — which is why a per-scan
  dark stack taken at the live settings always takes precedence over the
  TOML constant, and why the constant must be re-measured after a mode
  change. Both a plain std and a drift-immune consecutive-difference std
  are reported; they should agree (darks do not drift).
* Dark median: the median pixel value of the same frames — the electronic
  offset (bias + any dark signal, ~200 counts/px). It carries NO shot
  noise (it is not light), which is exactly why the Thompson bound
  subtracts it from the fitted background (dark_counts) instead of
  anyone ever subtracting it from the DATA — frames are always fitted raw
  (user rule 2026-08-20).

USAGE
    python measure_read_noise_and_dark.py darks1.pkl [darks2.pkl ...]

Each file: an AxialScan (or list) whose frames are CLOSED-SHUTTER
exposures, ideally several files at different exposure times (the ladder —
lets this script verify the no-dark-current claim).
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


def analyze_dark_stack(stack: np.ndarray) -> dict:
    plain_std = np.std(stack, axis=0, ddof=1)
    cdiff_std = np.std(np.diff(stack, axis=0), axis=0, ddof=1) / np.sqrt(2.0)
    return {
        "read_noise_plain": float(np.median(plain_std)),
        "read_noise_cdiff": float(np.median(cdiff_std)),
        "dark_median": float(np.median(stack)),
        "n_frames": stack.shape[0],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+",
                    help="AxialScan .pkl/.h5 files of closed-shutter frames")
    args = ap.parse_args(argv)

    rows = []
    for path in args.files:
        for scan in load_scans(path):
            stack = np.stack([np.asarray(m.frame_andor, dtype=float)
                              for m in scan.measurements])
            r = analyze_dark_stack(stack)
            r["exposure_s"] = float(scan.system_state.andor_camera_info.exposure)
            r["id"] = f"{path}:{scan.id}"
            rows.append(r)
            print(f"{r['id']}: exposure {r['exposure_s']:.3f} s, "
                  f"read noise {r['read_noise_plain']:.3f} counts "
                  f"(cdiff {r['read_noise_cdiff']:.3f}), "
                  f"dark median {r['dark_median']:.1f} counts/px, "
                  f"n = {r['n_frames']}")

    if not rows:
        print("No usable results.")
        return 1

    rn = np.array([r["read_noise_plain"] for r in rows])
    dm = np.array([r["dark_median"] for r in rows])
    exp = np.array([r["exposure_s"] for r in rows])

    print("\n==== Summary ====")
    print(f"read noise  = {np.median(rn):.3f} counts rms (median over "
          f"{rn.size} stacks)")
    print(f"dark median = {np.median(dm):.1f} counts/px")
    if np.unique(exp).size > 1:
        slope = np.polyfit(exp, dm, 1)[0]
        print(f"dark-current check: level slope {slope:+.2f} counts/px/s "
              f"over the ladder (expect ~0 — no dark current)")
        rn_slope = np.polyfit(exp, rn, 1)[0]
        print(f"read-noise-vs-exposure check: {rn_slope:+.3f} counts/s "
              f"(expect ~0 — the noise lives in the readout)")
    else:
        print("Only one exposure — take a ladder to verify no dark current.")

    print("\nPaste into ccd_characteristics.toml [ccd]:")
    print(f"read_noise_counts = {np.median(rn):.2f}")
    print(f"dark_median_counts = {np.median(dm):.1f}")
    print('read_noise_measured = "<date>"')
    print('dark_median_measured = "<date>"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
