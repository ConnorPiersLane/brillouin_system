"""Background model (prm0- vs prm1-style) and fit-window beta on the 2026-07-30 corneas.

The 7-30 files store only Lorentzian-family calibration polynomials (no raw
calibration frames), so the true pixel-response prm0/prm1 cannot be used here --
that pairing is the -168 MHz model-mixing trap. What CAN be varied inside the
calibration-consistent NA model (na_gauss_lorentzian_window, session D, n=1.376)
is exactly what distinguishes prm0 from prm1:

    background  flat_per_peak   (prm0-style: one constant per peak window)
                linear_per_peak (prm1-style: constant + slope per peak window)
    beta        3, 4, 5         (fit-window half-width = round(beta x HWHM))

For every combination this script refits ALL sweep-style scans, applies the
same gates as plot_cornea_depth_20260730 (|fwd-bwd| <= 50 um, fit valid), and
reports mean / sd / L-R split / residual rms. It also draws the fitted spectra
of one representative gated frame per subject.

Usage:
    PYTHONPATH=src python src/simple_plotting/cornea_fit_model_comparison_20260730.py <out_dir>
"""
import copy
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_cornea_depth_20260730 import (  # noqa: E402
    DATA, FITTING_MODEL, MAX_LR_DISAGREE, MAX_PLANE_GAP_UM, PLAUSIBLE,
    SESSION_D_MM, fit_scan, load_scans, planes, session_fitter,
)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

FILES = ["selina_50.h5", "selina_100um.h5", "yuxuan_50um.h5", "yuxuan_100um.h5"]
SHOW_FRAMES = {"Selina": "selina_100um.h5", "Yuxuan": "yuxuan_100um.h5"}

BACKGROUNDS = {"prm0-style (flat/peak)": "flat_per_peak",
               "prm1-style (linear/peak)": "linear_per_peak"}
BETAS = [3.0, 4.0, 5.0]
BG_COLORS = {"prm0-style (flat/peak)": "#D55E00", "prm1-style (linear/peak)": "#0072B2"}


def make_fitter(background: str, beta: float):
    sf = session_fitter()          # NA model, session D, n, pinned reference
    cfg = copy.deepcopy(sf.sample_config)
    cfg.background = background
    cfg.beta = beta
    sf.update_sample_config(cfg)
    return sf


def residual_rms(fs):
    m = fs.mask_for_fitting
    if fs.fitted_spectrum is None or m is None or not np.any(m):
        return np.nan
    return float(np.sqrt(np.mean((fs.sline[m] - fs.fitted_spectrum[m]) ** 2)))


def run_combo(scans_by_file: dict, sf):
    """Same gates as the depth figure; returns kept rows + all-frame rms."""
    shifts, splits, rmss = [], [], []
    for fn, scans in scans_by_file.items():
        for scan in scans:
            fz, bz = planes(scan)
            gap = (fz - bz) if (fz is not None and bz is not None) else np.nan
            if not (np.isfinite(gap) and abs(gap) <= MAX_PLANE_GAP_UM):
                continue
            s = fit_scan(scan, sf)[0]
            if not s.fitted_spectrum.is_success:
                continue
            a = s.analyzed_shifts
            shift = a.freq_shift_peak_distance_ghz
            left, right = a.freq_shift_left_peak_ghz, a.freq_shift_right_peak_ghz
            if shift is None or left is None or right is None:
                continue
            shift, left, right = float(shift), float(left), float(right)
            if not (abs(left - right) < MAX_LR_DISAGREE
                    and PLAUSIBLE[0] < shift < PLAUSIBLE[1]):
                continue
            shifts.append(shift)
            splits.append((right - left) * 1000.0)
            rmss.append(residual_rms(s.fitted_spectrum))
    return np.array(shifts), np.array(splits), np.array(rmss)


def first_kept_scan(scans, sf):
    """First scan that passes BOTH gates (plane pair AND a valid sample fit)."""
    for scan in scans:
        fz, bz = planes(scan)
        if not (fz is not None and bz is not None
                and abs(fz - bz) <= MAX_PLANE_GAP_UM):
            continue
        s = fit_scan(scan, sf)[0]
        if not s.fitted_spectrum.is_success:
            continue
        a = s.analyzed_shifts
        if a.freq_shift_peak_distance_ghz is None or a.freq_shift_left_peak_ghz is None:
            continue
        if (abs(a.freq_shift_left_peak_ghz - a.freq_shift_right_peak_ghz) < MAX_LR_DISAGREE
                and PLAUSIBLE[0] < a.freq_shift_peak_distance_ghz < PLAUSIBLE[1]):
            return scan
    return None


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    scans_by_file = {fn: load_scans(DATA / fn) for fn in FILES}

    # ---------- summary table over every gated sweep point ----------
    print(f"model {FITTING_MODEL}, D = {SESSION_D_MM} mm; gates identical to the "
          f"depth figure (|fwd-bwd| <= {MAX_PLANE_GAP_UM:.0f} um + fit valid)\n")
    header = (f"{'background':>26}{'beta':>6}{'n':>5}{'mean [GHz]':>12}{'sd [MHz]':>10}"
              f"{'L-R [MHz]':>12}{'sd(L-R)':>9}{'rms [cts]':>11}")
    print(header)
    results = {}
    for bg_label, bg in BACKGROUNDS.items():
        for beta in BETAS:
            sf = make_fitter(bg, beta)
            shifts, splits, rmss = run_combo(scans_by_file, sf)
            results[(bg_label, beta)] = (shifts, splits, rmss)
            print(f"{bg_label:>26}{beta:>6.0f}{len(shifts):>5}"
                  f"{shifts.mean():>12.4f}{shifts.std(ddof=1) * 1000:>10.1f}"
                  f"{splits.mean():>12.1f}{splits.std(ddof=1):>9.1f}"
                  f"{np.nanmean(rmss):>11.1f}")

    # ---------- fitted spectra of one representative gated frame ----------
    fig, axes = plt.subplots(2, len(BETAS), figsize=(13.5, 7.6), sharex="row")
    for r, (person, fn) in enumerate(SHOW_FRAMES.items()):
        scan = first_kept_scan(scans_by_file[fn], make_fitter("linear_per_peak", 3.0))
        for c, beta in enumerate(BETAS):
            ax = axes[r, c]
            for bg_label, bg in BACKGROUNDS.items():
                sf = make_fitter(bg, beta)
                s = fit_scan(scan, sf)[0]
                fs = s.fitted_spectrum
                col = BG_COLORS[bg_label]
                shift = s.analyzed_shifts.freq_shift_peak_distance_ghz
                ax.plot(fs.x_fit_refined, fs.y_fit_refined, "-", color=col, lw=1.5,
                        label=f"{bg_label.split(' ')[0]}: {shift:.4f} GHz, "
                              f"rms {residual_rms(fs):.0f}")
                if bg == "linear_per_peak":   # window pixels identical for both bgs
                    m = fs.mask_for_fitting
                    ax.plot(fs.x_pixels[~m], fs.sline[~m], "o", color="0.75", ms=4)
                    ax.plot(fs.x_pixels[m], fs.sline[m], "o", color="k", ms=4.5,
                            label=f"pixels in window ({int(m.sum())})")
            lo, hi = np.nonzero(fs.mask_for_fitting)[0][[0, -1]]
            ax.set_xlim(fs.x_pixels[max(lo - 6, 0)], fs.x_pixels[min(hi + 6, len(fs.x_pixels) - 1)])
            ax.set_title(f"{person}, scan i={scan.i}  —  β = {beta:.0f}", fontsize=10)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc="upper right")
            if c == 0:
                ax.set_ylabel("counts (summed rows)")
            if r == 1:
                ax.set_xlabel("pixel")

    fig.suptitle("Fitted cornea spectra — background model and fit window\n"
                 f"{FITTING_MODEL}, D = {SESSION_D_MM} mm; filled points = pixels "
                 "inside the β-window the fit actually uses", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "cornea_fit_model_comparison_20260730.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n-> {out}")
