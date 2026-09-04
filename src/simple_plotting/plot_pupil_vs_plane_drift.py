"""Is the cornea-surface z drift axial eye motion, or the eye moving laterally?

Cross-plots the two things the patient-movement GUI records simultaneously in
`cornea_track_*.json`:

  - the reflection-plane pair estimate (Zaber lens z, um)  -- what the Brillouin
    depth axis actually rides on
  - the eye tracker's pupil centre (center_ref_mm, x/y/z in mm, ~4 Hz)

Everything is converted to um and zeroed at its first sample, so all four traces
share ONE axis and are directly comparable -- no second y-scale.

Left column: all four displacements vs time.
Right column: pupil z vs plane z, with least-squares slope and Pearson r. A slope
near +1 (or -1) with high |r| means the plane drift IS bulk axial eye motion; a
flat/scattered cloud means it is not.

Usage:
    PYTHONPATH=src python src/simple_plotting/plot_pupil_vs_plane_drift.py <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2026-7-27")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

C_PLANE, C_X, C_Y, C_Z = "#009600", "#0072B2", "#D55E00", "#7A3B9A"
MARK_X, MARK_Y, MARK_Z = "o", "s", "^"
PEOPLE = ["connor", "zuriel"]


def load(path: Path):
    j = json.loads(path.read_text())
    est = j["estimates"]
    pup = [p for p in j.get("pupil_track", []) if p.get("center_ref_mm") is not None]
    if len(est) < 3 or len(pup) < 3:
        return None

    t0 = min(p["t_perf"] for p in j["points"])
    te = np.array([e["t_perf"] - t0 for e in est], float)
    ze = np.array([e["z_um"] for e in est], float)

    tp = np.array([p["t_perf"] - t0 for p in pup], float)
    xyz = np.array([p["center_ref_mm"] for p in pup], float) * 1000.0   # mm -> um

    # The stereo depth estimate occasionally fails outright (Zuriel 2026-07-27 has
    # single-sample jumps of 3.6-11.3 mm against a 17-30 um median step). Those are
    # dropped frames, not eye motion, and a handful of them destroys any
    # correlation. Reject the whole sample -- a bad depth means a bad detection,
    # so x and y are not trustworthy either. Real motion here is a few hundred um,
    # so a MAD gate floored at 1 mm cannot remove anything genuine.
    keep = np.ones(xyz.shape[0], bool)
    for k in range(3):
        c = xyz[:, k]
        mad = float(np.median(np.abs(c - np.median(c))))
        keep &= np.abs(c - np.median(c)) <= max(1000.0, 6.0 * 1.4826 * mad)
    n_drop = int((~keep).sum())
    tp, xyz = tp[keep], xyz[keep]
    if tp.size < 3:
        return None

    # Pupil is sampled ~4x faster than the pair estimate; put it on the estimate grid.
    inside = (te >= tp.min()) & (te <= tp.max())
    te, ze = te[inside], ze[inside]
    if te.size < 3:
        return None
    pup_i = np.column_stack([np.interp(te, tp, xyz[:, k]) for k in range(3)])

    return {"t": te, "plane": ze - ze[0], "pupil": pup_i - pup_i[0],
            "n_drop": n_drop, "n_pup": keep.size, "name": path.stem[13:]}


def plot_person(person: str):
    files = sorted((DATA / person).glob("cornea_track_*.json"))
    tracks = [t for t in (load(f) for f in files) if t is not None]
    if not tracks:
        print(f"{person}: nothing plottable")
        return

    fig, axes = plt.subplots(len(tracks), 4, figsize=(16, 2.7 * len(tracks) + 1.2),
                             gridspec_kw={"width_ratios": [2.6, 1, 1, 1]})
    axes = np.atleast_2d(axes)

    print(f"\n=== {person} — pupil vs plane drift ===")
    print(f"{'track':>14}{'n':>4}{'dropped':>9}{'plane span':>11}"
          f"{'  x: slope     r':>18}{'  y: slope     r':>18}{'  z: slope     r':>18}")

    for row, tr in enumerate(tracks):
        axl = axes[row, 0]
        t, plane, pup = tr["t"], tr["plane"], tr["pupil"]

        axl.axhline(0, color="0.75", lw=1.0)
        axl.plot(t, plane, "-", color=C_PLANE, lw=2.4, label="reflection plane (pair est.)")
        for k, (c, mk, lbl) in enumerate([(C_X, MARK_X, "pupil x"),
                                          (C_Y, MARK_Y, "pupil y"),
                                          (C_Z, MARK_Z, "pupil z")]):
            axl.plot(t, pup[:, k], "-", marker=mk, ms=3.5, lw=1.2, color=c,
                     alpha=0.9, label=lbl)
        axl.set_ylabel("displacement (µm)", fontsize=8.5)
        axl.set_title(tr["name"], fontsize=8.5, loc="left")
        axl.grid(alpha=0.3)

        # Same scatter for each pupil axis, so x/y act as controls for z.
        stats = []
        for k, (c, lbl) in enumerate([(C_X, "x"), (C_Y, "y"), (C_Z, "z")]):
            axr = axes[row, k + 1]
            pk = pup[:, k]
            slope, r = np.nan, np.nan
            if np.std(plane) > 1e-9 and np.std(pk) > 1e-9:
                slope = float(np.polyfit(plane, pk, 1)[0])
                r = float(np.corrcoef(plane, pk)[0, 1])
            stats.append((slope, r))
            axr.scatter(plane, pk, s=20, color=c, alpha=0.8, edgecolor="none")
            if np.isfinite(slope):
                xs = np.array([plane.min(), plane.max()])
                axr.plot(xs, np.polyval([slope, np.mean(pk) - slope * np.mean(plane)], xs),
                         "-", color="0.35", lw=1.3)
            # Equal x and y spans, so the drawn slope is visually honest.
            lim = max(np.ptp(plane), np.ptp(pk)) * 0.6 + 20
            axr.set_xlim(plane.mean() - lim, plane.mean() + lim)
            axr.set_ylim(pk.mean() - lim, pk.mean() + lim)
            axr.set_xlabel("plane drift (µm)", fontsize=8)
            axr.set_ylabel(f"pupil {lbl} drift (µm)", fontsize=8)
            axr.set_title(f"{lbl}:  slope {slope:+.2f}   r {r:+.2f}", fontsize=8.5, loc="left")
            axr.grid(alpha=0.3)
            axr.tick_params(labelsize=7.5)

        print(f"{tr['name']:>14}{t.size:4d}{tr['n_drop']:>4}/{tr['n_pup']:<4}"
              f"{np.ptp(plane):10.0f}µm"
              + "".join(f"{s:11.2f}{r:7.2f}" for s, r in stats))

    axes[0, 0].legend(fontsize=7.5, ncol=4, loc="upper left")
    axes[-1, 0].set_xlabel("time since tracking started (s)")
    fig.suptitle(f"{person.capitalize()} — pupil-centre motion vs. reflection-plane "
                 f"drift, 2026-07-27\nall traces zeroed at their first sample; "
                 f"one shared µm axis", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = OUT / f"pupil_vs_plane_{person}_20260727.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for p in PEOPLE:
        plot_person(p)
