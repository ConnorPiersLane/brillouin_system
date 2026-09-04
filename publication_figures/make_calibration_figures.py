"""Publication-ready (Optica style) calibration figures.

Data: 2026-7-9 afternoon session, calibration.h5 (401 EOM sweep points, 4-8 GHz,
frames + production 2lorentzian_window fits stored).

Outputs (600 dpi PNG, single-column 8.6 cm width):
  fig_calibration_spectrum.png  - EOM sideband spectrum at 5.5 GHz, x-axis in GHz
  fig_calibration_curve.png     - distance calibration: EOM freq vs peak separation + residuals
  fig_calibration_combined.png  - both as panels (a)/(b)

Run:  .venv\\Scripts\\python.exe publication_figures\\make_calibration_figures.py
      (PYTHONPATH must include src/)
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from brillouin_system.calibration.calibration import CalibrationCalculator, calibrate
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    dict_to_dataclass_tree,
    load_dict_from_hdf5,
)

CAL_PATH = (
    r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data"
    r"\2026-7-9\afternoon\calibration.h5"
)
OUT_DIR = Path(__file__).resolve().parent
TARGET_FREQ_GHZ = 5.5
POLY_DEGREE = 2

# --- Optica-style rc ---------------------------------------------------------
CM = 1 / 2.54
SINGLE_COL = 8.6 * CM  # Optica single-column figure width

COL_DATA = "#3b3b3b"   # measured points: near-black neutral
COL_FIT = "#0C5DA5"    # fit line: single print-safe blue
COL_ORDER = "#9a9a9a"  # elastic-order markers: recessive gray
COL_ACCENT = "#B02425" # residuals / annotation accent (dark red)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})


def load_calibration():
    data = dict_to_dataclass_tree(load_dict_from_hdf5(CAL_PATH), known_classes)
    params = calibrate(data, POLY_DEGREE)
    calc = CalibrationCalculator(params)
    return data, params, calc


def pick_point(data, target_ghz):
    freqs = np.array([m.set_freq_ghz for m in data.measured_freqs])
    idx = int(np.argmin(np.abs(freqs - target_ghz)))
    return data.measured_freqs[idx]


def _mask_runs(mask):
    """Contiguous True runs of a boolean mask as lists of indices."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, splits)


# --- panel (a): sideband spectrum at ~5.5 GHz, GHz axis ----------------------
def draw_spectrum(ax, meas, calc):
    fit = meas.cali_meas_points[0].fitting_results
    anchors = calc.elastic_anchors()

    # px -> GHz: frequency measured from the left elastic order (freq_left_peak
    # poly, monotonic over the window); the next order lands at the FSR.
    def px_to_ghz(px):
        return calc.freq_left_peak(px)

    x_data = px_to_ghz(fit.x_pixels)
    fsr = px_to_ghz(anchors.rayleigh_right_px)

    kcounts = 1e-3
    # the _window model fits only masked regions around each peak: draw the
    # fit curve over those windows, not across the whole spectrum
    label = "Lorentzian fit"
    for run in _mask_runs(fit.mask_for_fitting):
        px_lo, px_hi = fit.x_pixels[run[0]] - 0.5, fit.x_pixels[run[-1]] + 0.5
        seg = (fit.x_fit_refined >= px_lo) & (fit.x_fit_refined <= px_hi)
        ax.plot(
            px_to_ghz(fit.x_fit_refined[seg]), fit.y_fit_refined[seg] * kcounts,
            color=COL_FIT, lw=1.0, zorder=3, label=label,
        )
        label = None
    ax.plot(
        x_data, fit.sline * kcounts,
        ls="none", marker="o", ms=2.6, mfc="white", mec=COL_DATA, mew=0.7,
        zorder=4, label="Measured",
    )

    for x0 in (px_to_ghz(anchors.rayleigh_left_px), fsr):
        ax.axvline(x0, color=COL_ORDER, lw=0.7, ls=(0, (4, 3)), zorder=1)

    ymax = fit.sline.max() * kcounts
    ax.set_ylim(0, ymax * 1.32)
    ax.set_xlim(px_to_ghz(fit.x_pixels[0]) - 0.5, px_to_ghz(fit.x_pixels[-1]) + 0.5)

    # elastic-order labels (kept below the legend to avoid collisions)
    ax.text(
        0.25, ymax * 0.86, "order $m$",
        ha="left", va="top", fontsize=7, color="#6d6d6d",
    )
    ax.text(
        fsr - 0.25, ymax * 0.86, "order $m{+}1$",
        ha="right", va="top", fontsize=7, color="#6d6d6d",
    )

    # sideband annotations: each peak sits nu_EOM from its own elastic order
    left_pk = px_to_ghz(fit.left_peak_center_px)
    right_pk = px_to_ghz(fit.right_peak_center_px)
    y_arrow = ymax * 0.55
    for x0, x1 in ((0, left_pk), (fsr, right_pk)):
        ax.annotate(
            "", xy=(x1, y_arrow), xytext=(x0, y_arrow),
            arrowprops=dict(
                arrowstyle="->", color=COL_DATA, lw=0.7, shrinkA=0, shrinkB=2,
            ),
        )
    ax.text(
        left_pk / 2, y_arrow + ymax * 0.04,
        r"$+\nu_{\mathrm{EOM}}$", ha="center", va="bottom", fontsize=7.5,
    )
    ax.text(
        (fsr + right_pk) / 2, y_arrow + ymax * 0.04,
        r"$-\nu_{\mathrm{EOM}}$", ha="center", va="bottom", fontsize=7.5,
    )
    ax.text(
        (left_pk + right_pk) / 2, ymax * 1.05,
        rf"$\nu_{{\mathrm{{EOM}}}} = {meas.set_freq_ghz:.1f}\,$GHz",
        ha="center", va="top", fontsize=7.5,
    )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"Intensity ($10^{3}$ counts)")
    ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0.4)


# --- panel (b): distance calibration curve + residuals -----------------------
def draw_calibration_curve(ax_main, ax_res, data, calc):
    d_px, f_ghz = [], []
    for m in data.measured_freqs:
        fr = m.cali_meas_points[0].fitting_results
        if fr.is_success and fr.inter_peak_distance is not None:
            d_px.append(fr.inter_peak_distance)
            f_ghz.append(m.set_freq_ghz)
    d_px = np.asarray(d_px)
    f_ghz = np.asarray(f_ghz)

    d_grid = np.linspace(d_px.min(), d_px.max(), 300)
    ax_main.plot(
        d_grid, calc.freq_peak_distance(d_grid),
        color=COL_FIT, lw=1.0, zorder=3,
        label="Quadratic fit",
    )
    ax_main.plot(
        d_px, f_ghz,
        ls="none", marker="o", ms=2.0, mfc="white", mec=COL_DATA, mew=0.55,
        zorder=2, label=f"EOM sweep ({len(d_px)} pts)",
    )
    ax_main.set_ylabel("EOM frequency (GHz)")
    ax_main.legend(loc="upper right", handlelength=1.4, borderaxespad=0.4)
    ax_main.tick_params(labelbottom=False)

    res_mhz = (f_ghz - calc.freq_peak_distance(d_px)) * 1e3
    rms = float(np.sqrt(np.mean(res_mhz**2)))
    ax_res.axhline(0, color=COL_ORDER, lw=0.6, zorder=1)
    ax_res.plot(
        d_px, res_mhz,
        ls="none", marker="o", ms=1.8, mfc=COL_ACCENT, mec="none", alpha=0.6,
        zorder=2,
    )
    lim = np.ceil(np.max(np.abs(res_mhz)) / 5) * 5
    ax_res.set_ylim(-lim, lim)
    ax_res.set_xlabel("Sideband separation (pixels)")
    ax_res.set_ylabel("Resid.\n(MHz)")
    ax_res.text(
        0.02, 0.92, f"RMS = {rms:.1f} MHz",
        transform=ax_res.transAxes, ha="left", va="top", fontsize=7,
    )
    return rms


def panel_label(ax, letter):
    ax.text(
        -0.16, 1.02, f"({letter})", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=9, fontweight="bold",
    )


def main():
    data, _, calc = load_calibration()
    meas = pick_point(data, TARGET_FREQ_GHZ)

    # figure 1: spectrum only
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 5.8 * CM))
    draw_spectrum(ax, meas, calc)
    fig.savefig(OUT_DIR / "fig_calibration_spectrum.png")
    plt.close(fig)

    # figure 2: calibration curve + residuals
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(SINGLE_COL, 6.8 * CM), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.08},
    )
    rms = draw_calibration_curve(ax1, ax2, data, calc)
    fig.savefig(OUT_DIR / "fig_calibration_curve.png")
    plt.close(fig)

    # combined (a)/(b)
    fig = plt.figure(figsize=(SINGLE_COL, 12.2 * CM))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[2.4, 2.2, 0.85], hspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2], sharex=ax_b)
    draw_spectrum(ax_a, meas, calc)
    draw_calibration_curve(ax_b, ax_c, data, calc)
    # sharex puts the gap from hspace between (b) and residuals; tighten it
    pos_b, pos_c = ax_b.get_position(), ax_c.get_position()
    ax_c.set_position([pos_c.x0, pos_b.y0 - pos_c.height * 1.12,
                       pos_c.width, pos_c.height])
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    fig.savefig(OUT_DIR / "fig_calibration_combined.png")
    plt.close(fig)

    print(f"saved 3 figures to {OUT_DIR}")
    print(f"spectrum point: set_freq = {meas.set_freq_ghz} GHz")
    print(f"distance-chain residual RMS = {rms:.2f} MHz")


if __name__ == "__main__":
    main()
