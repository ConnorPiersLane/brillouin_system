"""Fitted spectra behind the 2026-08-21 sweep-scan analysis.

Two figures, both showing only frames that passed BOTH gates (crossings agree
within 50 um, fit valid), each panel annotated with its NA-corrected shift:

  cornea_sweep_spectra_20260821_connor_zuriel.png
      every kept frame for Connor and Zuriel.

  cornea_sweep_spectra_20260821_jimmy.png
      Jimmy sorted by depth, up to N_PER_DEPTH frames per commanded depth, so
      the stroma -> posterior surface -> aqueous progression is visible. The
      question this answers: the frames reading 5.49-5.60 GHz near 300-400 um
      are either genuine partial-volume (focus straddling the posterior
      surface) or bad fits. A partial-volume frame still shows two clean
      peaks, just closer together; a bad fit does not.

Usage:
    PYTHONPATH=src python src/simple_plotting/plot_cornea_sweep_spectra_20260821.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brillouin_system.calibration.calibration import calibration_calculator_for_scan
from brillouin_system.spectrum_fitting.na_lineshape import na_mean_shift_ratio

from plot_cornea_sweep_20260821 import (  # noqa: E402
    AQUEOUS_BELOW_GHZ, DATA, MAX_LR_DISAGREE, MAX_PLANE_GAP_UM, PLAUSIBLE,
    SESSION_D_MM, crossing_z, load_scans, na_corrected, session_fitter,
)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
NCOLS = 6
N_PER_DEPTH = 3


def kept_frames(filename: str, sf, ratio: float):
    """(target, depth, shift, FittedSpectrum) for every frame passing both gates."""
    out = []
    for scan in load_scans(DATA / filename):
        calc = calibration_calculator_for_scan(
            scan.calibration_data, scan.calibration_params, sf)
        ss = scan.system_state
        target = (float(scan.sweep_config.target_depth_um)
                  if scan.sweep_config is not None else np.nan)
        for cycle in (scan.sweep_cycles or []):
            z_in, z_out = crossing_z(cycle.reflection_in), crossing_z(cycle.reflection_out)
            mi = cycle.measurement_index
            if mi is None or z_in is None or z_out is None:
                continue
            if abs(z_in - z_out) > MAX_PLANE_GAP_UM:
                continue
            m = scan.measurements[mi]
            px, sline = sf.get_px_sline_from_image(m.frame_andor.copy())
            fit = sf.fit(px=px, sline=sline, is_reference_mode=ss.is_reference_mode)
            if not fit.is_success:
                continue
            a = na_corrected(calc.analyze(fit), ratio)
            s, l, r = (a.freq_shift_peak_distance_ghz,
                       a.freq_shift_left_peak_ghz, a.freq_shift_right_peak_ghz)
            if s is None or l is None or r is None:
                continue
            s, l, r = float(s), float(l), float(r)
            if not (abs(l - r) < MAX_LR_DISAGREE and PLAUSIBLE[0] < s < PLAUSIBLE[1]):
                continue
            depth = float(m.lens_zaber_position) - 0.5 * (z_in + z_out)
            out.append(dict(scan_i=int(scan.i), cycle=int(cycle.cycle_index),
                            target=target, depth=depth, shift=s, fit=fit))
    return out


def draw_grid(panels, title, outfile, label_of):
    nrows = int(np.ceil(len(panels) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(2.35 * NCOLS, 1.95 * nrows))
    axes = np.atleast_2d(axes)
    for k, p in enumerate(panels):
        ax = axes[k // NCOLS, k % NCOLS]
        fs = p["fit"]
        m = fs.mask_for_fitting
        col = "#C1662F" if p["shift"] < AQUEOUS_BELOW_GHZ else "#0072B2"
        ax.plot(fs.x_pixels[~m], fs.sline[~m], "o", color="0.78", ms=2.4)
        ax.plot(fs.x_pixels[m], fs.sline[m], "o", color="0.15", ms=2.8)
        ax.plot(fs.x_fit_refined, fs.y_fit_refined, "-", color=col, lw=1.3)
        lo, hi = np.nonzero(m)[0][[0, -1]]
        ax.set_xlim(fs.x_pixels[max(lo - 6, 0)],
                    fs.x_pixels[min(hi + 6, len(fs.x_pixels) - 1)])
        pad = 0.10 * (fs.sline.max() - fs.sline.min())
        ax.set_ylim(fs.sline.min() - pad, fs.sline.max() + 3.4 * pad)
        ax.set_title(label_of(p), fontsize=7.5, color=col, pad=2)
        ax.text(0.03, 0.96, f"{p['shift']:.4f} GHz\nd = {p['depth']:.0f} µm",
                transform=ax.transAxes, va="top", ha="left", fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        if k % NCOLS:
            ax.set_yticklabels([])
    for k in range(len(panels), nrows * NCOLS):
        axes[k // NCOLS, k % NCOLS].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.supxlabel("pixel", fontsize=9)
    fig.supylabel("counts (summed rows)", fontsize=9)
    fig.tight_layout(rect=(0.012, 0.012, 1, 0.94))
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"-> {outfile}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sf = session_fitter()
    ratio = na_mean_shift_ratio(sf.sample_config)

    common = (f"black = pixels in the fit window; blue = cornea, orange = aqueous "
              f"(< {AQUEOUS_BELOW_GHZ} GHz)\nprm1 vs lorentzian_x_psf calibration "
              f"re-fitted per scan; NA 0.42 post-hoc, D = {SESSION_D_MM} mm, n = 1.376")

    panels = []
    for who, fn in (("Connor", "connor.h5"), ("Zuriel", "zuriel.h5")):
        for p in kept_frames(fn, sf, ratio):
            p["who"] = who
            panels.append(p)
    panels.sort(key=lambda p: (p["who"], p["target"], p["depth"]))
    draw_grid(panels,
              "Connor and Zuriel — every kept sweep frame, 2026-08-21\n" + common,
              OUT / "cornea_sweep_spectra_20260821_connor_zuriel.png",
              lambda p: f"{p['who']}  i={p['scan_i']} c{p['cycle']}")

    jimmy = kept_frames("jimmy.h5", sf, ratio)
    chosen = []
    for target in sorted({p["target"] for p in jimmy}):
        at = sorted((p for p in jimmy if p["target"] == target),
                    key=lambda p: p["shift"])
        # Spread the sample across the shift range at this depth, so a mixed
        # depth shows both its cornea and its aqueous frames.
        idx = np.unique(np.linspace(0, len(at) - 1, min(N_PER_DEPTH, len(at))).astype(int))
        chosen += [at[i] for i in idx]
    chosen.sort(key=lambda p: (p["target"], p["shift"]))
    draw_grid(chosen,
              f"Jimmy — kept sweep frames by commanded depth (up to {N_PER_DEPTH} per "
              f"depth, spanning that depth's shift range), 2026-08-21\n" + common,
              OUT / "cornea_sweep_spectra_20260821_jimmy.png",
              lambda p: f"target {p['target']:.0f} µm  i={p['scan_i']} c{p['cycle']}")
