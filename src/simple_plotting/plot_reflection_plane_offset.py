"""Diagnostic: where does the reflection-finder plane sit relative to depth 0?

Depth 0 in the axial-profile figures is the FIRST VALID FIT of each scan (our
working definition of the corneal front surface). The reflection finder gives an
independent estimate of the same surface. This plots both on the same axis so the
disagreement is visible per scan.

Usage:
    PYTHONPATH=src python src/simple_plotting/plot_reflection_plane_offset.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_axial_depth_profile import (  # noqa: E402
    DATA, MIN_DEPTH_SCAN_POINTS, PALETTE, MARKERS, N_CORNEA, SESSION_D_MM,
    N_SAMPLE, FITTING_MODEL, PLAUSIBLE, MAX_LR_DISAGREE,
    load_scans, session_fitter, fit_scan,
)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
FAILED_Y = 5.7   # rejected frames parked here; only their x position means anything

PEOPLE = [("Connor", "connorAxial.h5"),
          ("Selina", "selinaAxialUpdated.h5"),
          ("Zuriel", "zurielAxial.h5")]


def extract_with_z0(scan):
    """Same validity rule as the main figure, but also return z0 and the planes."""
    spectra = fit_scan(scan, session_fitter())
    z = np.array([m.lens_zaber_position for m in scan.measurements], float)

    def col(getter):
        return np.array([getter(s) if s.fitted_spectrum.is_success else np.nan
                         for s in spectra], float)

    shift = col(lambda s: s.analyzed_shifts.freq_shift_peak_distance_ghz)
    left = col(lambda s: s.analyzed_shifts.freq_shift_left_peak_ghz)
    right = col(lambda s: s.analyzed_shifts.freq_shift_right_peak_ghz)
    err = col(lambda s: s.theoretical_precisions.distance_total_mhz) / 1000.0

    good = (np.isfinite(shift) & np.isfinite(left) & np.isfinite(right)
            & (np.abs(left - right) < MAX_LR_DISAGREE)
            & (shift > PLAUSIBLE[0]) & (shift < PLAUSIBLE[1]))
    if not good.any():
        return None

    z0 = z[np.argmax(good)]
    fwd = (scan.reflection_result_forwards.event_z_um
           if scan.reflection_result_forwards else None)
    bwd = (scan.reflection_result_backwards.event_z_um
           if scan.reflection_result_backwards else None)
    z_off = (scan.reflection_result_forwards.z_offset_um
             if scan.reflection_result_forwards else None)
    return z[good] - z0, shift[good], err[good], z0, fwd, bwd, z[~good] - z0, z_off


def plot_person(PERSON: str, FILENAME: str):
    scans = [s for s in load_scans(DATA / FILENAME)
             if len(s.measurements) >= MIN_DEPTH_SCAN_POINTS]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    print(f"=== {PERSON}: reflection plane relative to depth 0 (first valid fit) ===")
    print("The first SAMPLE never sits at the commanded z_offset: hi_backend.py:531 does")
    print("move_rel(step) at the TOP of the loop, so sample 0 lands at z_offset + 1 step.")
    print("'start' is the measured z_first - fwd; 'expect' is z_offset + step. They match.")
    print("Whatever is left ('rejected') is frames the validity rule threw out.\n")
    print(f"{'scan':>6}{'z0 [um]':>10}{'step':>7}{'fwd-z0':>9}{'start':>9}"
          f"{'expect':>9}{'rejected':>11}{'bwd-z0':>9}{'fwd-bwd':>9}")

    for k, scan in enumerate(scans):
        res = extract_with_z0(scan)
        if res is None:
            continue
        depth, shift, err, z0, fwd, bwd, depth_bad, z_off = res
        color, marker = PALETTE[k % len(PALETTE)], MARKERS[k % len(MARKERS)]

        ax.errorbar(depth, shift, yerr=err, lw=1.5, capsize=2.5, alpha=0.9, ms=6,
                    color=color, marker=marker, label=f"scan {scan.i}")
        # Rejected frames carry no meaningful shift (they are elastic reflection),
        # so park them on a common line -- only their x position is informative.
        if depth_bad.size:
            ax.plot(depth_bad, np.full(depth_bad.size, FAILED_Y), "x",
                    color=color, ms=9, mew=2.0, zorder=6)

        z_first = float(scan.measurements[0].lens_zaber_position)
        step = float(np.median(np.diff([m.lens_zaber_position for m in scan.measurements])))
        d_f = (fwd - z0) if fwd is not None else np.nan
        d_b = (bwd - z0) if bwd is not None else np.nan
        gap = (fwd - bwd) if (fwd is not None and bwd is not None) else np.nan
        start_gap = (z_first - fwd) if fwd is not None else np.nan
        print(f"{scan.i:>6}{z0:10.1f}{step:7.0f}{d_f:9.1f}{start_gap:9.1f}"
              f"{z_off + step:9.1f}{z_first - z0:11.1f}{d_b:9.1f}{gap:9.1f}")

        if fwd is not None:
            ax.axvline(d_f, color=color, ls="--", lw=1.4, alpha=0.85)
            ax.annotate(f"fwd {d_f:+.0f}", xy=(d_f, 0.97), xycoords=("data", "axes fraction"),
                        rotation=90, va="top", ha="right", fontsize=7.5, color=color)
        if bwd is not None:
            ax.axvline(d_b, color=color, ls=":", lw=1.4, alpha=0.85)
            ax.annotate(f"bwd {d_b:+.0f}", xy=(d_b, 0.97), xycoords=("data", "axes fraction"),
                        rotation=90, va="top", ha="right", fontsize=7.5, color=color)

    ax.axvline(0, color="0.25", lw=1.8, zorder=1)
    ax.annotate("depth 0 = first valid fit", xy=(0, 0.03), xycoords=("data", "axes fraction"),
                rotation=90, va="bottom", ha="right", fontsize=8, color="0.25")

    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.35", ls="--", lw=1.4),
                plt.Line2D([], [], color="0.35", ls=":", lw=1.4),
                plt.Line2D([], [], color="0.25", lw=1.8),
                plt.Line2D([], [], color="0.35", ls="none", marker="x", ms=9, mew=2.0)]
    labels += ["reflection plane, forward", "reflection plane, backward", "depth 0",
               f"rejected frame (parked at {FAILED_Y}; x position only)"]
    ax.legend(handles, labels, fontsize=8, loc="lower left")

    ax.set_xlabel("Depth relative to first valid fit — Zaber lens travel (µm)")
    ax.set_ylabel("Brillouin shift (GHz)")
    ax.set_title(f"{PERSON} — reflection-finder plane vs. the first-valid-fit surface, "
                 f"2026-07-23\nnegative = finder puts the surface BEFORE the first usable "
                 f"spectrum", fontsize=10)
    ax.grid(alpha=0.3)
    secax = ax.secondary_xaxis("top", functions=(lambda d: d * N_CORNEA,
                                                lambda d: d / N_CORNEA))
    secax.set_xlabel(f"approx. focus depth in tissue (µm, ×n={N_CORNEA})", fontsize=9)
    ax.text(0.995, 0.02, f"{FITTING_MODEL}, D = {SESSION_D_MM} mm, n = {N_SAMPLE}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.45")

    fig.tight_layout()
    out = OUT / f"reflection_plane_offset_{PERSON.lower()}_20260723.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for person, filename in PEOPLE:
        plot_person(person, filename)
