"""Cornea surface tracking over time, 2026-07-27 (one figure per person).

Reads the `cornea_track_*.json` files written by the patient-movement GUI
(pm_backend.save_track). Reproduces the GUI's live strip chart:

  up crossings    blue  up-triangles    (sweep moving one way)
  down crossings  orange down-triangles (sweep moving the other way)
  pair estimate   green line            (up/down averaged -> latency-bias free)

y = 0 is that track's INITIALLY found reflection plane
(`meta.reflection_plane_z_um`), so every panel shows movement relative to where
the surface was when tracking started. x = seconds since the first pass.

Usage:
    PYTHONPATH=src python src/simple_plotting/plot_cornea_tracks.py <out_dir>
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

# The GUI's own colours (pm_frontend._group_strip_chart), kept for familiarity.
C_UP, C_DOWN, C_EST = "#3C82D2", "#E08020", "#009600"

PEOPLE = ["connor", "zuriel"]


def load_track(path: Path):
    j = json.loads(path.read_text())
    plane = float(j["meta"]["reflection_plane_z_um"])
    pts, est = j["points"], j.get("estimates", [])
    t0 = min(p["t_perf"] for p in pts)

    def arr(rows, tkey="t_perf", zkey="z_um"):
        t = np.array([r[tkey] - t0 for r in rows], float)
        z = np.array([r[zkey] for r in rows], float) - plane
        return t, z

    found = [p for p in pts if p["found"] and p["z_um"] is not None]
    up = [p for p in found if p["direction"] == "up"]
    down = [p for p in found if p["direction"] == "down"]
    amp = float(j["meta"]["tracking_config"]["sweep_amplitude_um"])
    return {
        "up": arr(up), "down": arr(down), "est": arr(est),
        "n_pts": len(pts), "n_found": len(found), "plane": plane,
        "amp": amp, "name": path.stem.replace("cornea_track_", ""),
    }


def plot_person(person: str):
    files = sorted((DATA / person).glob("cornea_track_*.json"))
    tracks = [load_track(f) for f in files]

    fig, axes = plt.subplots(len(tracks), 1, sharey=True,
                             figsize=(9.5, 2.5 * len(tracks) + 1.2))
    axes = np.atleast_1d(axes)

    print(f"\n=== {person} — {len(tracks)} cornea track(s) ===")
    for ax, tr in zip(axes, tracks):
        ax.axhline(0, color="0.35", lw=1.4, zorder=1)
        ax.plot(*tr["up"], ls="none", marker="^", ms=6, color=C_UP,
                alpha=0.85, label="up crossings")
        ax.plot(*tr["down"], ls="none", marker="v", ms=6, color=C_DOWN,
                alpha=0.85, label="down crossings")
        ax.plot(*tr["est"], "-o", ms=4, lw=1.8, color=C_EST,
                label="pair estimate (bias-free)")

        t_est, z_est = tr["est"]
        drift = (z_est[-1] - z_est[0]) if z_est.size >= 2 else np.nan
        ax.set_title(f"{tr['name']}   plane {tr['plane']:.0f} µm   "
                     f"{tr['n_found']}/{tr['n_pts']} passes found   "
                     f"sweep ±{tr['amp']:.0f} µm", fontsize=8.5, loc="left")
        ax.grid(alpha=0.3)
        ax.set_ylabel("surface z − initial\nplane (µm)", fontsize=8.5)

        print(f"  {tr['name']}: plane {tr['plane']:8.1f}  found {tr['n_found']:2d}/"
              f"{tr['n_pts']:2d}  pair-est range {z_est.min():+7.1f}..{z_est.max():+7.1f} µm"
              f"  span {np.ptp(z_est):6.1f}  net drift {drift:+7.1f}")

    axes[0].legend(fontsize=8, ncol=3, loc="upper left")
    axes[-1].set_xlabel("time since tracking started (s)")
    fig.suptitle(f"{person.capitalize()} — cornea surface tracking, 2026-07-27\n"
                 f"y = 0 is the initially found reflection plane",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT / f"cornea_track_{person}_20260727.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for p in PEOPLE:
        plot_person(p)
