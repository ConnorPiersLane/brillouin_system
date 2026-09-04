"""Sweep-style single-frame scans vs. conventional axial scans, 2026-07-30.

Both methods, same session, same eye, same fitter (na_gauss_lorentzian_window,
session D = 6.241 mm from the day's own water bracket, n = 1.376), so the only
difference is HOW depth was registered:

  sweep style   one frame per scan; the reflection plane is found going IN and
                again going OUT around that single frame (~2 s apart). Depth =
                lens z - pair average. Gate: |fwd - bwd| <= 50 um.
  axial scan    the plane is found ONCE, then 10-20 frames are stepped over
                ~18 s, and the plane is re-found at the end. Depth = lens z -
                forward plane; the fwd-bwd gap measures how far the eye moved
                during the whole scan.

Usage:
    PYTHONPATH=src python src/simple_plotting/compare_sweep_vs_axial_20260730.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_cornea_depth_20260730 import (  # noqa: E402
    DATA, MAX_LR_DISAGREE, MAX_PLANE_GAP_UM, N_CORNEA, PLAUSIBLE, SESSION_D_MM,
    collect, fit_scan, load_scans, planes, session_fitter,
)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

SWEEP_FILES = {
    "Selina": ["selina_50.h5", "selina_100um.h5"],
    "Yuxuan": ["yuxuan_50um.h5", "yuxuan_100um.h5"],
}
AXIAL_FILES = {
    "Selina": "selina_axialscan.h5",
    "Yuxuan": "yuxuan_axialscan.h5",
}

AXIAL_PALETTE = ["#009E73", "#7A3B9A", "#B8860B", "#555555", "#CC79A7", "#56B4E9"]
SWEEP_PALETTE = ["#0072B2", "#D55E00"]
SWEEP_MARKERS = ["o", "s"]
PLATEAU_UM = 150.0   # lens travel window over which the stroma should be flat


def axial_rows(scan, sf):
    """Depth (vs forward plane), shift and validity for every frame of one scan."""
    fz, bz = planes(scan)
    z = np.array([m.lens_zaber_position for m in scan.measurements], float)
    spectra = fit_scan(scan, sf)

    def col(getter):
        return np.array([getter(s) if s.fitted_spectrum.is_success else np.nan
                         for s in spectra], float)

    shift = col(lambda s: s.analyzed_shifts.freq_shift_peak_distance_ghz)
    left = col(lambda s: s.analyzed_shifts.freq_shift_left_peak_ghz)
    right = col(lambda s: s.analyzed_shifts.freq_shift_right_peak_ghz)
    good = (np.isfinite(shift) & np.isfinite(left) & np.isfinite(right)
            & (np.abs(left - right) < MAX_LR_DISAGREE)
            & (shift > PLAUSIBLE[0]) & (shift < PLAUSIBLE[1]))
    depth = z - fz if fz is not None else z - z[0]
    gap = (fz - bz) if (fz is not None and bz is not None) else np.nan
    return depth, shift, good, gap


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sf = session_fitter()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)

    for ax, person in zip(axes, SWEEP_FILES):
        # --- conventional axial scans: connected lines, one color per scan
        print(f"\n=== {person}: conventional axial scans ===")
        for k, scan in enumerate(load_scans(DATA / AXIAL_FILES[person])):
            depth, shift, good, gap = axial_rows(scan, sf)
            n_good = int(good.sum())
            gap_lbl = f"{gap:+.0f} um" if np.isfinite(gap) else "backward find FAILED"
            plateau = shift[good & (depth <= PLATEAU_UM)]
            plateau_lbl = (f", plateau sd {plateau.std(ddof=1)*1000:.0f} MHz"
                           if plateau.size > 1 else "")
            print(f"  scan i={scan.i}: {n_good}/{len(shift)} valid fits, "
                  f"eye moved (fwd-bwd) {gap_lbl}{plateau_lbl}")
            if n_good == 0:
                continue
            c = AXIAL_PALETTE[k % len(AXIAL_PALETTE)]
            ax.plot(depth[good], shift[good], "-", color=c, lw=1.2, alpha=0.55, zorder=2)
            ax.plot(depth[good], shift[good], "^", color=c, ms=5, alpha=0.75, zorder=2,
                    label=f"axial scan {scan.i}  (plane moved {gap_lbl})")

        # --- sweep-style points that pass the strict gate, on top
        for k, fn in enumerate(SWEEP_FILES[person]):
            kept = [r for r in collect(fn, sf) if r["keep"]]
            if not kept:
                continue
            d = np.array([r["depth"] for r in kept])
            y = np.array([r["shift"] for r in kept])
            e = np.array([r["err"] for r in kept])
            sd = y.std(ddof=1) * 1000 if len(y) > 1 else float("nan")
            ax.errorbar(d, y, yerr=e, ls="none", marker=SWEEP_MARKERS[k], ms=8,
                        capsize=2.5, lw=1.3, color=SWEEP_PALETTE[k], zorder=5,
                        markeredgecolor="black", markeredgewidth=0.6,
                        label=f"sweep {fn.replace('.h5','')}  "
                              f"(n={len(y)}, sd {sd:.0f} MHz)")

        ax.set_title(person, fontsize=11)
        ax.set_xlabel("Depth past the surface — Zaber lens travel (µm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower left")
        secax = ax.secondary_xaxis("top", functions=(lambda d: d * N_CORNEA,
                                                     lambda d: d / N_CORNEA))
        secax.set_xlabel(f"approx. depth in tissue (µm, ×n={N_CORNEA})", fontsize=8)

    axes[0].set_ylabel("Brillouin shift (GHz)")
    fig.suptitle(
        "Sweep-style scans (plane re-found around every frame, |fwd−bwd| ≤ 50 µm)\n"
        "vs. conventional axial scans (plane found once, depth axis drifts with the eye) "
        "— 2026-07-30", fontsize=11)
    fig.text(0.995, 0.005,
             f"na_gauss_lorentzian_window vs. lorentzian calibration, "
             f"D = {SESSION_D_MM} mm (session water bracket), n = 1.376; "
             f"axial depth = lens z − forward plane",
             ha="right", va="bottom", fontsize=7, color="0.45")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    out = OUT / "sweep_vs_axial_20260730.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n-> {out}")
