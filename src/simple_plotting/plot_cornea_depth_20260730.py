"""Brillouin shift vs. depth in the cornea for the 2026-07-30 subject scans.

Each scan in these files is ONE frame taken at a commanded offset past the
surface found by the reflection finder on the way IN (forward). The finder also
runs on the way OUT (backward), giving a second, independent estimate of the
same surface. That pair is the depth-registration quality check:

    gate:  both planes found AND |forward - backward| <= 50 um
    depth: lens_z - (forward + backward)/2

i.e. the pair average is the surface, which cancels the sweep latency bias
(2026-07-30 sweep validation: pair-averaged residual ~ -3 um at 2 mm/s, vs a
one-directional bias of several um). Points where the two planes disagree by
more than the gate are thrown out, not re-registered.

x = Zaber lens travel past that surface (the MEASURED axis), with a secondary
    top axis converting to approximate focus depth in tissue (x n = 1.376).
y = Brillouin shift from the inter-peak distance, NA-corrected.

D = 6.241 mm is solved on THIS session's own two-NA water bracket
(water_na042_na014.h5, 4 scans x 50 frames): na042 and na014 both land on
f180 = 5.0704 GHz with a residual gap of -0.000 MHz (raw two-NA gap -14.1 MHz).
Re-solve with
    python src/simple_plotting/solve_session_D_from_water.py <water .h5>

The calibration is fitted with REFERENCE_MODEL below, NOT with whatever the live
TOML happens to hold -- see the comment there. Mixing families moves every shift
by ~3 MHz.

Usage:
    PYTHONPATH=src python src/simple_plotting/plot_cornea_depth_20260730.py <out_dir>
"""
import copy
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.human_interface_measurements import AxialScan, AnalyzedSpectrum
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import (
    calculate_photon_counts_from_fitted_spectrum,
)
from brillouin_system.spectrum_fitting.helpers.subtract_background import (
    subtract_background, subtract_darknoise,
)
from brillouin_system.spectrum_fitting.spectrum_analyzer import SpectrumAnalyzer
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter, model_requires_anchors

DATA = Path(r"C:\Users\cplan\Dropbox (Personal)\Boston\Data\2026-7-30")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

N_CORNEA = 1.376
PLAUSIBLE = (4.0, 8.0)      # GHz, physically possible cornea Brillouin shift
MAX_LR_DISAGREE = 0.3       # GHz; left/right must agree or the fit caught noise
MAX_PLANE_GAP_UM = 50.0     # the strict depth-registration gate

# Solved on THIS session's own two-NA water bracket (see docstring). Prefer a
# same-day bracket over the 5.8 house default; here they differ by 0.44 mm,
# i.e. +1.7 MHz on every point (common-mode -- it cannot change any shape).
SESSION_D_MM = 6.241
N_SAMPLE = 1.376
FITTING_MODEL = "na_gauss_lorentzian_window"
# The calibration must be fitted in the SAME lineshape family as the samples --
# the fitter refuses to mix pixel-response with plain-Lorentzian centres. The
# live TOML tracks whatever the GUI was last set to (prm1/pixel_response at the
# time of writing), so pin the reference here rather than inherit it.
REFERENCE_MODEL = "lorentzian"

PALETTE = ["#0072B2", "#D55E00"]
MARKERS = ["o", "s"]

PEOPLE = {
    "Selina": ["selina_50.h5", "selina_100um.h5"],
    "Yuxuan": ["yuxuan_50um.h5", "yuxuan_100um.h5"],
}


def load_scans(path: Path) -> list[AxialScan]:
    obj = dict_to_dataclass_tree(load_dict_from_hdf5(str(path)), known_classes)
    return obj if isinstance(obj, list) else [obj]


def session_fitter() -> SpectrumFitter:
    """Fitter pinned to ONE stated D for the whole session (never refit per point)."""
    sf = SpectrumFitter()
    cfg = copy.deepcopy(sf.sample_config)
    cfg.fitting_model = FITTING_MODEL
    cfg.na_beam_diameter_mm = SESSION_D_MM
    cfg.na_n_sample = N_SAMPLE
    sf.update_sample_config(cfg)

    ref = copy.deepcopy(sf.reference_config)
    ref.fitting_model = REFERENCE_MODEL
    sf.update_reference_config(ref)
    return sf


def fit_scan(scan: AxialScan, sf: SpectrumFitter) -> list[AnalyzedSpectrum]:
    calc = CalibrationCalculator(parameters=scan.calibration_params)
    analyzer = SpectrumAnalyzer(calibration_calculator=calc)
    ss = scan.system_state
    cam = ss.andor_camera_info

    anchors = None
    if not ss.is_reference_mode and model_requires_anchors(sf.sample_config.fitting_model):
        anchors = calc.elastic_anchors()

    out = []
    for measurement in scan.measurements:
        frame = measurement.frame_andor.copy()
        frame = (subtract_background(frame=frame, bg_frame=ss.bg_image)
                 if ss.is_do_bg_subtraction_active
                 else subtract_darknoise(frame=frame, darknoise_frame=ss.dark_image))

        px, sline = sf.get_px_sline_from_image(frame)
        fit = sf.fit(px=px, sline=sline, is_reference_mode=ss.is_reference_mode, anchors=anchors)
        photons = calculate_photon_counts_from_fitted_spectrum(
            fs=fit, preamp_gain=cam.preamp_gain, emccd_gain=cam.gain)
        theo = analyzer.theoretical_precision(
            fs=fit, photons=photons,
            bg_frame_std=ss.bg_image.std_image if ss.is_do_bg_subtraction_active else None,
            preamp_gain=cam.preamp_gain, emccd_gain=cam.gain)
        out.append(AnalyzedSpectrum(fitted_spectrum=fit,
                                    analyzed_shifts=analyzer.analyze_spectrum(fitting=fit),
                                    photons=photons, theoretical_precisions=theo))
    return out


def planes(scan: AxialScan):
    """(forward, backward) surface z in um, None where the finder did not fire."""
    f = scan.reflection_result_forwards
    b = scan.reflection_result_backwards
    fz = float(f.event_z_um) if (f is not None and f.found and f.event_z_um is not None) else None
    bz = float(b.event_z_um) if (b is not None and b.found and b.event_z_um is not None) else None
    return fz, bz


def collect(filename: str, sf: SpectrumFitter):
    """One row per scan: depth, shift, err, and why it was kept or dropped."""
    rows = []
    for scan in load_scans(DATA / filename):
        fz, bz = planes(scan)
        gap = (fz - bz) if (fz is not None and bz is not None) else np.nan
        z_lens = float(scan.measurements[0].lens_zaber_position)

        spectra = fit_scan(scan, sf)
        s = spectra[0]
        ok_fit = s.fitted_spectrum.is_success
        shift = s.analyzed_shifts.freq_shift_peak_distance_ghz if ok_fit else np.nan
        left = s.analyzed_shifts.freq_shift_left_peak_ghz if ok_fit else np.nan
        right = s.analyzed_shifts.freq_shift_right_peak_ghz if ok_fit else np.nan
        err = (s.theoretical_precisions.distance_total_mhz / 1000.0) if ok_fit else np.nan
        shift = np.nan if shift is None else float(shift)
        left = np.nan if left is None else float(left)
        right = np.nan if right is None else float(right)

        pass_plane = np.isfinite(gap) and abs(gap) <= MAX_PLANE_GAP_UM
        pass_fit = (np.isfinite(shift) and np.isfinite(left) and np.isfinite(right)
                    and abs(left - right) < MAX_LR_DISAGREE
                    and PLAUSIBLE[0] < shift < PLAUSIBLE[1])
        depth = (z_lens - 0.5 * (fz + bz)) if np.isfinite(gap) else np.nan

        if not np.isfinite(gap):
            why = "no backward plane"
        elif not pass_plane:
            why = f"planes disagree {gap:+.0f} um"
        elif not pass_fit:
            if not (np.isfinite(shift) and np.isfinite(left) and np.isfinite(right)):
                why = "fit failed"
            elif abs(left - right) < MAX_LR_DISAGREE:
                why = f"shift {shift:.3f} GHz outside {PLAUSIBLE}"
            else:
                why = f"L-R disagree {(left-right)*1000:+.0f} MHz"
        else:
            why = "kept"
        rows.append(dict(i=scan.i, gap=gap, depth=depth, shift=shift, err=err,
                         fit_ok=pass_fit, keep=pass_plane and pass_fit, why=why))
    return rows


def plot_person(ax, name: str, files: list[str], sf: SpectrumFitter):
    print(f"\n=== {name} ===")
    all_kept = []
    for k, fn in enumerate(files):
        rows = collect(fn, sf)
        kept = [r for r in rows if r["keep"]]
        all_kept += kept
        n_plane = sum(1 for r in rows if np.isfinite(r["gap"]) and abs(r["gap"]) <= MAX_PLANE_GAP_UM)
        print(f"  {fn}: {len(rows)} scans -> {n_plane} pass the {MAX_PLANE_GAP_UM:.0f} um "
              f"plane gate -> {len(kept)} also fit")
        for r in rows:
            if not r["keep"]:
                print(f"      drop i={r['i']:>3}: {r['why']}")

        if not kept:
            continue
        d = np.array([r["depth"] for r in kept])
        y = np.array([r["shift"] for r in kept])
        e = np.array([r["err"] for r in kept])
        spread = (f"± {y.std(ddof=1)*1000:.1f} MHz sd" if len(y) > 1 else "single point")
        ax.errorbar(d, y, yerr=e, ls="none", marker=MARKERS[k % len(MARKERS)], ms=7,
                    capsize=2.5, lw=1.3, alpha=0.9, color=PALETTE[k % len(PALETTE)],
                    label=f"{fn.replace('.h5', '')}  (n={len(kept)}, "
                          f"{y.mean():.4f} GHz {spread})")
        sem = (y.std(ddof=1)/np.sqrt(len(y))*1000) if len(y) > 1 else float("nan")
        print(f"      depth {d.min():.0f}-{d.max():.0f} um, "
              f"shift {y.mean():.4f} GHz {spread} (sem {sem:.1f})")

    if len(all_kept) >= 3:
        d = np.array([r["depth"] for r in all_kept])
        y = np.array([r["shift"] for r in all_kept])
        b, a = np.polyfit(d, y, 1)
        # slope uncertainty from the residual scatter
        resid = y - (a + b * d)
        se_b = (resid.std(ddof=2) / np.sqrt(((d - d.mean()) ** 2).sum()))
        xs = np.linspace(d.min(), d.max(), 50)
        ax.plot(xs, a + b * xs, "-", color="0.35", lw=1.4, zorder=1,
                label=f"linear: {b*1e3*1e3:+.1f} ± {se_b*1e6:.1f} MHz/mm travel")
        print(f"  pooled slope {b*1e6:+.1f} ± {se_b*1e6:.1f} MHz per mm of travel "
              f"({b*1e6/N_CORNEA:+.1f} MHz per mm in tissue), n={len(all_kept)}")

    ax.set_title(f"{name}", fontsize=11)
    ax.set_xlabel("Depth past the surface — Zaber lens travel (µm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="best")
    secax = ax.secondary_xaxis("top", functions=(lambda d: d * N_CORNEA,
                                                lambda d: d / N_CORNEA))
    secax.set_xlabel(f"approx. depth in tissue (µm, ×n={N_CORNEA})", fontsize=8)


def gate_sensitivity(sf: SpectrumFitter):
    """Does the strict gate change the answer, or only the sample size?

    Re-applies the plane gate at several thresholds on the SAME fits. If the
    mean moves far more than its own SEM as the gate loosens, the gate is
    selecting on shift (bad); if it only buys precision, 50 um is just strict.
    """
    print("\n=== plane-gate sensitivity (same fits, gate varied) ===")
    print(f"{'file':>18}{'gate':>8}{'n':>5}{'mean [GHz]':>13}{'sd [MHz]':>11}{'sem':>8}")
    for name, files in PEOPLE.items():
        for fn in files:
            rows = collect(fn, sf)
            for gate in (50.0, 75.0, 100.0, 1e9):
                sel = [r for r in rows
                       if r["fit_ok"] and np.isfinite(r["gap"]) and abs(r["gap"]) <= gate]
                if len(sel) < 2:
                    continue
                y = np.array([r["shift"] for r in sel])
                lbl = f"{gate:.0f}" if gate < 1e8 else "none"
                print(f"{fn.replace('.h5',''):>18}{lbl:>8}{len(y):>5}{y.mean():13.4f}"
                      f"{y.std(ddof=1)*1000:11.1f}{y.std(ddof=1)/np.sqrt(len(y))*1000:8.1f}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sf = session_fitter()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, (name, files) in zip(axes, PEOPLE.items()):
        plot_person(ax, name, files, sf)
    axes[0].set_ylabel("Brillouin shift (GHz)")
    fig.suptitle("Cornea Brillouin shift vs. depth — 2026-07-30\n"
                 f"strict gate: forward and backward reflection planes agree within "
                 f"{MAX_PLANE_GAP_UM:.0f} µm; depth measured from their average",
                 fontsize=11)
    fig.text(0.995, 0.005,
             f"{FITTING_MODEL} vs. {REFERENCE_MODEL} calibration, D = {SESSION_D_MM} mm "
             f"(solved on this session's water bracket, residual gap 0.0 MHz), "
             f"n = {N_SAMPLE}",
             ha="right", va="bottom", fontsize=7, color="0.45")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = OUT / "cornea_depth_20260730.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n-> {out}")

    gate_sensitivity(sf)
