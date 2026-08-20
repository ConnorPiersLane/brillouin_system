"""Every fitted cornea spectrum behind the 2026-07-30 sweep-scan analysis.

One small panel per KEPT point (both gates: |fwd-bwd| <= 50 um plane pair AND a
valid fit), showing the summed-row spectrum, the pixels inside the beta-window
the fit uses, the production fit (na_gauss_lorentzian_window = NA042 kernel with
the session D, prm1-style per-peak linear background, beta = 3), and the
NA-corrected Brillouin shift of that frame. The NA correction is inside the fit
kernel, so the annotated numbers are the corrected values plotted in the depth
figure.

Usage:
    PYTHONPATH=src python src/simple_plotting/plot_cornea_fitted_spectra_20260730.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_cornea_depth_20260730 import (  # noqa: E402
    DATA, MAX_LR_DISAGREE, MAX_PLANE_GAP_UM, PLAUSIBLE, SESSION_D_MM,
    fit_scan, load_scans, planes, session_fitter,
)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

FILES = ["selina_50.h5", "selina_100um.h5", "yuxuan_50um.h5", "yuxuan_100um.h5"]
FILE_COLORS = {"selina_50": "#0072B2", "selina_100um": "#D55E00",
               "yuxuan_50um": "#009E73", "yuxuan_100um": "#7A3B9A"}

NCOLS = 6


def kept_points(sf):
    """(label, color, depth, AnalyzedSpectrum) for every point the figure kept."""
    out = []
    for fn in FILES:
        short = fn.replace(".h5", "")
        for scan in load_scans(DATA / fn):
            fz, bz = planes(scan)
            if not (fz is not None and bz is not None
                    and abs(fz - bz) <= MAX_PLANE_GAP_UM):
                continue
            s = fit_scan(scan, sf)[0]
            if not s.fitted_spectrum.is_success:
                continue
            a = s.analyzed_shifts
            if (a.freq_shift_peak_distance_ghz is None
                    or a.freq_shift_left_peak_ghz is None
                    or a.freq_shift_right_peak_ghz is None):
                continue
            if not (abs(a.freq_shift_left_peak_ghz - a.freq_shift_right_peak_ghz)
                    < MAX_LR_DISAGREE
                    and PLAUSIBLE[0] < a.freq_shift_peak_distance_ghz < PLAUSIBLE[1]):
                continue
            depth = float(scan.measurements[0].lens_zaber_position) - 0.5 * (fz + bz)
            out.append((f"{short}  i={scan.i}", FILE_COLORS[short], depth, s))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sf = session_fitter()
    points = kept_points(sf)
    print(f"{len(points)} kept points")

    nrows = int(np.ceil(len(points) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(2.35 * NCOLS, 1.95 * nrows))
    axes = np.atleast_2d(axes)

    for k, (label, color, depth, s) in enumerate(points):
        ax = axes[k // NCOLS, k % NCOLS]
        fs = s.fitted_spectrum
        m = fs.mask_for_fitting
        ax.plot(fs.x_pixels[~m], fs.sline[~m], "o", color="0.78", ms=2.6)
        ax.plot(fs.x_pixels[m], fs.sline[m], "o", color="0.15", ms=3.0)
        ax.plot(fs.x_fit_refined, fs.y_fit_refined, "-", color=color, lw=1.3)

        lo, hi = np.nonzero(m)[0][[0, -1]]
        ax.set_xlim(fs.x_pixels[max(lo - 6, 0)],
                    fs.x_pixels[min(hi + 6, len(fs.x_pixels) - 1)])
        pad = 0.10 * (fs.sline.max() - fs.sline.min())
        ax.set_ylim(fs.sline.min() - pad, fs.sline.max() + 3.2 * pad)

        shift = s.analyzed_shifts.freq_shift_peak_distance_ghz
        sig = s.theoretical_precisions.distance_total_mhz
        ax.set_title(label, fontsize=7.5, color=color, pad=2)
        ax.text(0.03, 0.96, f"{shift:.4f} GHz\nd = {depth:.0f} µm  σ = {sig:.1f} MHz",
                transform=ax.transAxes, va="top", ha="left", fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        if k % NCOLS:
            ax.set_yticklabels([])

    for k in range(len(points), nrows * NCOLS):
        axes[k // NCOLS, k % NCOLS].axis("off")

    fig.suptitle(
        "All kept sweep-scan cornea spectra — 2026-07-30   "
        "(black = pixels in the fit window, curve = production fit)\n"
        f"annotated shifts are NA-corrected: na_gauss_lorentzian_window kernel, "
        f"NA 0.42, D = {SESSION_D_MM} mm (session water bracket), n = 1.376; "
        "per-peak linear background, β = 3", fontsize=10)
    fig.supxlabel("pixel", fontsize=9)
    fig.supylabel("counts (summed rows)", fontsize=9)
    fig.tight_layout(rect=(0.012, 0.012, 1, 0.94))
    out = OUT / "cornea_fitted_spectra_20260730.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")
