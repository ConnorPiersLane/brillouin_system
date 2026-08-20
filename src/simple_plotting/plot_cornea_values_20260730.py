"""Just the values: NA-corrected shifts of the valid 2026-07-30 sweep points.

Only the kept scans (plane pair within 50 um AND valid fit), plotted against
the scan index (acquisition order), one color per file, with the per-file
mean +/- sd band. Error bars are the per-frame shot-noise sigma.

Usage:
    PYTHONPATH=src python src/simple_plotting/plot_cornea_values_20260730.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_cornea_depth_20260730 import N_CORNEA, SESSION_D_MM, session_fitter  # noqa: E402
from plot_cornea_fitted_spectra_20260730 import FILES, FILE_COLORS, kept_points  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    points = kept_points(session_fitter())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    panel_of = {"selina": 0, "yuxuan": 1}

    for fn in FILES:
        short = fn.replace(".h5", "")
        rows = [(depth, s) for label, color, depth, s in points
                if label.startswith(short)]
        if not rows:
            continue
        ax = axes[panel_of[short.split("_")[0]]]
        d = np.array([depth for depth, _ in rows])
        y = np.array([float(s.analyzed_shifts.freq_shift_peak_distance_ghz)
                      for _, s in rows])
        c = FILE_COLORS[short]
        stats = (f"{y.mean():.4f} GHz ± {y.std(ddof=1)*1000:.1f} MHz sd"
                 if len(y) > 1 else f"{y[0]:.4f} GHz (single point)")
        # No per-frame shot-noise bars: the points run well above the photon
        # limit (see the paper's noise discussion), so the honest uncertainty is
        # the point-to-point sd -- carried by the shaded band, not per-point bars.
        ax.plot(d, y, ls="none", marker="o", ms=7, color=c,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"{short}  (n={len(y)}, {stats})")
        if len(y) > 1:
            ax.axhspan(y.mean() - y.std(ddof=1), y.mean() + y.std(ddof=1),
                       color=c, alpha=0.10, zorder=0)
            ax.axhline(y.mean(), color=c, lw=1.0, alpha=0.6, zorder=1)

    for name, ax in zip(["Selina", "Yuxuan"], axes):
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Depth past the surface — Zaber lens travel (µm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        secax = ax.secondary_xaxis("top", functions=(lambda d: d * N_CORNEA,
                                                     lambda d: d / N_CORNEA))
        secax.set_xlabel(f"optical depth in tissue (µm, ×n={N_CORNEA})", fontsize=9)
    axes[0].set_ylabel("Brillouin shift (GHz)")
    # Reserve an empty band at the bottom so the legends never sit on a point.
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    axes[0].set_ylim(lo - 0.30 * (hi - lo), hi)

    fig.suptitle("Valid sweep-scan points only — NA-corrected shift per frame, 2026-07-30\n"
                 "line and band = per-file mean ± sd (scan-to-scan scatter)",
                 fontsize=11)
    fig.text(0.995, 0.005,
             f"na_gauss_lorentzian_window, NA 0.42, D = {SESSION_D_MM} mm, n = 1.376; "
             "gates: |fwd−bwd| ≤ 50 µm, valid fit",
             ha="right", va="bottom", fontsize=7, color="0.45")
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    out = OUT / "cornea_values_20260730.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")
