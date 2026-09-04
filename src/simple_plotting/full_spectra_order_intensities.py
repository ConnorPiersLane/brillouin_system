"""How much light lands in etalon orders other than the main one? (PI request 2026-08-04)

Uses the full-chip "full_spectra" data taken 2026-07-31. For water (150 frames
pooled over 3 scans) it y-averages the CCD image, finds all Brillouin peaks,
and reports each peak's amplitude and integrated area relative to the main
order pair (px 97/124).

Result on this data: the other orders carry ~60% of the main pair's light
(by area), i.e. above the 50% threshold for pursuing the global multi-order fit.

Usage:
    PYTHONPATH=src python src/simple_plotting/full_spectra_order_intensities.py [out_dir]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DATA = Path(r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2026-7-31")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

MAIN_PAIR_PX = (97, 124)


def profile_from_frames(frames):
    """Mean over frames, then y-average -> 1D column profile, baseline removed."""
    stack = np.stack(frames).astype(float)
    img = stack.mean(axis=0)
    prof = img.mean(axis=0)
    return prof - np.median(prof), img


def peak_table(prof, prominence=None):
    if prominence is None:
        prominence = max(1.0, 0.02 * prof.max())
    idx, _ = find_peaks(prof, prominence=prominence, distance=5)
    return idx, prof[idx]


def main():
    with open(DATA / "full_spectra_water.pkl", "rb") as f:
        water = pickle.load(f)
    with open(DATA / "full_spectra_reference.pkl", "rb") as f:
        ref = pickle.load(f)

    water_frames = [m.frame_andor for s in water for m in s.measurements]
    prof_w, img_w = profile_from_frames(water_frames)

    ref_profiles = {}
    for s in ref:
        frames = [m.frame_andor for m in s.measurements]
        p, _ = profile_from_frames(frames)
        ref_profiles[s.id] = p

    # ---- intensity bookkeeping (water)
    idx, amps = peak_table(prof_w, prominence=1.5)
    half = 5
    areas = {i: prof_w[max(0, i - half):i + half + 1].sum() for i in idx}
    main_area = sum(areas[i] for i in idx if i in MAIN_PAIR_PX)
    other_area = sum(a for i, a in areas.items() if i not in MAIN_PAIR_PX)
    print("water peaks (px, amp, area, area % of main pair):")
    for i in idx:
        print(f"  {i:4d}  {prof_w[i]:7.2f}  {areas[i]:7.2f}  {areas[i] / main_area * 100:5.1f}%")
    print(f"light outside the main order: {other_area / main_area * 100:.1f}% of the main pair")

    # ---- figure
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].imshow(img_w, aspect="auto", cmap="inferno")
    axes[0].set_title("Water: mean CCD image (150 frames)")
    axes[0].set_ylabel("row (y)")

    axes[1].plot(prof_w, "k-", lw=1)
    axes[1].plot(idx, amps, "rv", ms=6)
    for i, a in zip(idx, amps):
        axes[1].annotate(f"{a / amps.max() * 100:.0f}%", (i, a), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[1].set_title("Water: y-averaged profile, peak amplitudes as % of max")
    axes[1].set_ylabel("counts")

    for name, p in sorted(ref_profiles.items()):
        axes[2].plot(p, lw=1, label=name)
    axes[2].set_title("Reference scans (EOM sidebands)")
    axes[2].set_xlabel("pixel (x)")
    axes[2].set_ylabel("counts")
    axes[2].legend(fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(OUT / "order_intensities.png", dpi=140)
    print("saved", OUT / "order_intensities.png")


if __name__ == "__main__":
    main()
