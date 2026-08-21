"""Build a packaged ReflectionBackground from a reflection-plane measurement.

Input: the session's reflection-plane .pkl (a list of scans, each with N
identical-exposure frames and its own EOM calibration) plus the session's
CLOSED-shutter dark .pkl for the bias level ("laser blocked" is not zero
light — the open-shutter darks carry a small elastic leak, measured
2026-08-19).

What it does:
1. keeps the scans with the expected frame count and drops intensity
   outliers (e.g. the deliberate 2x-intensity linearity check scan),
2. averages all kept frames into one bias-subtracted 2D frame (stored 2D so
   the row band can be re-selected at apply time — the y-alignment handling),
3. fits the session's own calibration sidebands with the production
   reference fitter and stores the (freq, left px, right px) points — the
   template's frequency anchor.

Usage:
    python -m brillouin_system.spectrum_fitting.build_reflection_background \
        <reflection_plane.pkl> <dark_closed_shutter.pkl> [-o out.npz]

The default output is the packaged production template path
(reflection_background_data/reflection_bg_2026-08-19_4pk.npz).
"""
from __future__ import annotations

import argparse
import datetime
import pickle
from pathlib import Path

import numpy as np

from brillouin_system.spectrum_fitting.reflection_background import (
    DEFAULT_REFLECTION_BG,
    ReflectionBackground,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


def _mean_frame(scan) -> np.ndarray:
    return np.mean(
        [np.asarray(m.frame_andor, dtype=float) for m in scan.measurements],
        axis=0,
    )


def _calibration_points(scan, fitter: SpectrumFitter):
    """(freqs, left px, right px) from a scan's own EOM calibration."""
    freqs, left, right = [], [], []
    for block in scan.calibration_data.measured_freqs:
        ls, rs = [], []
        for point in block.cali_meas_points:
            px, sline = fitter.get_px_sline_from_image(
                np.asarray(point.frame, dtype=float))
            fs = fitter.fit(px, sline, is_reference_mode=True)
            if fs.is_success:
                ls.append(float(fs.left_peak_center_px))
                rs.append(float(fs.right_peak_center_px))
        if ls:
            freqs.append(float(block.set_freq_ghz))
            left.append(float(np.mean(ls)))
            right.append(float(np.mean(rs)))
    return np.array(freqs), np.array(left), np.array(right)


def build_reflection_background(
    reflection_pkl: Path | str,
    dark_pkl: Path | str,
    frames_per_scan: int = 100,
    notes: str = "",
) -> ReflectionBackground:
    reflection_pkl = Path(reflection_pkl)
    dark_pkl = Path(dark_pkl)

    with open(dark_pkl, "rb") as f:
        dark_scans = pickle.load(f)
    bias = float(np.median(np.stack(
        [np.asarray(m.frame_andor, dtype=float)
         for m in dark_scans[0].measurements])))
    print(f"bias (closed shutter) = {bias:.2f} counts")

    with open(reflection_pkl, "rb") as f:
        scans = pickle.load(f)
    candidates = [s for s in scans if len(s.measurements) == frames_per_scan]
    if not candidates:
        raise ValueError(
            f"No scans with {frames_per_scan} frames in {reflection_pkl}."
        )

    # Drop intensity outliers (>1.5x / <0.5x the median total signal), such
    # as the 2x-intensity linearity-check scan of the 2026-08-19 session.
    means = [_mean_frame(s) for s in candidates]
    totals = np.array([float(np.sum(m - bias)) for m in means])
    ref = float(np.median(totals))
    keep = (totals > 0.5 * ref) & (totals < 1.5 * ref)
    for s, tot, k in zip(candidates, totals, keep):
        tag = "kept" if k else "DROPPED (intensity outlier)"
        print(f"  scan {getattr(s, 'id', '?')}: total {tot:.3e} "
              f"({tot / ref:.2f}x median) — {tag}")
    kept = [m for m, k in zip(means, keep) if k]
    kept_scans = [s for s, k in zip(candidates, keep) if k]

    frame = np.mean(kept, axis=0) - bias
    print(f"template frame: {frame.shape[0]}x{frame.shape[1]}, "
          f"{len(kept)} scans x {frames_per_scan} frames, "
          f"max {frame.max():.0f} counts")

    fitter = SpectrumFitter()
    fitter.auto_select_rows(np.stack(
        [np.asarray(p.frame, dtype=float)
         for b in kept_scans[0].calibration_data.measured_freqs
         for p in b.cali_meas_points]))
    freqs, left, right = _calibration_points(kept_scans[0], fitter)
    print(f"calibration: {len(freqs)} points, "
          f"{freqs.min():.2f}-{freqs.max():.2f} GHz")

    meta = {
        "source": str(reflection_pkl),
        "dark_source": str(dark_pkl),
        "bias_counts": bias,
        "n_scans": len(kept),
        "frames_per_scan": frames_per_scan,
        "built": datetime.date.today().isoformat(),
        "notes": notes,
    }
    return ReflectionBackground(
        frame=frame, cal_freqs=freqs, cal_left_px=left, cal_right_px=right,
        meta=meta,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reflection_pkl", type=Path)
    ap.add_argument("dark_pkl", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_REFLECTION_BG)
    ap.add_argument("--frames-per-scan", type=int, default=100)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    bg = build_reflection_background(
        args.reflection_pkl, args.dark_pkl,
        frames_per_scan=args.frames_per_scan, notes=args.notes,
    )
    bg.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
