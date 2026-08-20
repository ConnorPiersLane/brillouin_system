"""Brillouin shift vs. cornea depth for the 2026-07-23 axial scans.

One figure per person, with every axial scan in that person's file overlaid.

x = Zaber lens travel measured from the first valid measurement *of that scan*,
    which is our working definition of the corneal front surface. The reflection
    finder's plane is deliberately ignored here.
    A secondary top axis converts travel to approximate focus depth in tissue
    (~n x travel, n = 1.376), following docs/na_project_status.md.
y = Brillouin shift (GHz) from the inter-peak distance, i.e. the calibration
    reference currently configured ("distance").

Usage:
    PYTHONPATH=src python src/simple_plotting/plot_axial_depth_profile.py <out_dir>
"""
import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.human_interface_measurements import AxialScan, AnalyzedSpectrum
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree
from brillouin_system.spectrum_fitting.noise_analysis import (
    PixelCountsAndPhotons, theoretical_precision,
)
from brillouin_system.spectrum_fitting.helpers.subtract_darknoise import subtract_darknoise
from brillouin_system.spectrum_fitting.na_lineshape import na_mean_shift_ratio
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

DATA = Path(r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2026-7-23")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

N_CORNEA = 1.376
PLAUSIBLE = (4.0, 8.0)      # GHz, physically possible cornea Brillouin shift
MAX_LR_DISAGREE = 0.3       # GHz; left/right must agree or the fit caught noise
MIN_DEPTH_SCAN_POINTS = 10  # anything shorter is a repeat-at-one-position file

# Categorical palette (scans are identities, not a magnitude ramp), assigned in
# fixed order and never cycled. Validated for colour-vision deficiency on a white
# surface; the blue/purple pair sits at the 6-8 dE floor, so marker shape carries
# the identity as well and neither series depends on colour alone.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#7A3B9A"]
MARKERS = ["o", "s", "^", "D"]

# House default when a session has NO water bracket at all and none is close in
# time. Pooled mean of every self-consistency-solved D to date, collapsed per
# SESSION so same-day brackets do not get extra weight (7-9 5.24, 7-13 6.80,
# 7-14 5.46, 7-15 5.67, 7-17 6.12, 7-20 5.50, 7-27 5.52 -> mean 5.76, sd 0.53).
# Rounding to 5.8 is free on RMS (1.78 MHz either way) and slightly better in the
# worst case (3.6 vs 3.8 MHz, both on 7-13). Prefer a real bracket over this.
DEFAULT_D_MM = 5.8

# --- NA model parameters: ONE value for the whole session, never refit per point.
# D (na_beam_diameter_mm) is the per-session coupling knob, calibrated on a water
# bracket. 2026-07-23 has no bracket of its own, so this uses the house default
# above -- Connor's standing choice, so every no-bracket session is treated alike.
# For reference, the nearest bracket (2026-07-27 `water_na014_na042.h5`, 4 days
# later) solves to D = 5.52 +/- 0.15 mm (water f180 = 5.0692 GHz, two-NA residual
# 0.00 MHz, raw gap -11.8 MHz). Using 5.8 instead of that costs 1.0 MHz here.
# Measured sensitivity: +3.6 MHz per +1 mm of D, i.e. 5.7 MHz across the whole
# 5.24-6.80 mm range ever observed. D is common-mode: it slides every point
# together and cannot create or move the posterior step.
SESSION_D_MM = DEFAULT_D_MM
N_SAMPLE = 1.376        # cornea (the live TOML sits at 1.33 = water; worth 0.9 MHz)
# The NA lineshape models were removed 2026-08-20: fit a plain windowed
# Lorentzian and DIVIDE the shifts by the post-hoc scalar <cos(v/2)>
# (na_mean_shift_ratio, Gaussian coupling weight) — validated equivalent.
FITTING_MODEL = "lorentzian_window"

# Only files holding real depth scans. connor40/connor50 are 3x repeats at a
# single z (reflection-plane searches), not axial profiles.
PEOPLE = {
    "Connor": "connorAxial.h5",
    "Selina": "selinaAxialUpdated.h5",
    "Zuriel": "zurielAxial.h5",
}


def load_scans(path: Path) -> list[AxialScan]:
    obj = dict_to_dataclass_tree(load_dict_from_hdf5(str(path)), known_classes)
    return obj if isinstance(obj, list) else [obj]


def session_fitter() -> SpectrumFitter:
    """Fitter pinned to ONE stated D for the whole session (never refit per point).

    Overrides the live TOML in memory only -- the config file is not written.
    """
    sf = SpectrumFitter()
    cfg = copy.deepcopy(sf.sample_config)
    cfg.fitting_model = FITTING_MODEL
    cfg.na_beam_diameter_mm = SESSION_D_MM
    cfg.na_n_sample = N_SAMPLE
    sf.update_sample_config(cfg)
    return sf


def na_corrected(shifts, ratio):
    """Divide the three shifts by the post-hoc NA ratio (widths untouched)."""
    def d(v):
        return None if v is None else v / ratio
    return replace(
        shifts,
        freq_shift_left_peak_ghz=d(shifts.freq_shift_left_peak_ghz),
        freq_shift_right_peak_ghz=d(shifts.freq_shift_right_peak_ghz),
        freq_shift_peak_distance_ghz=d(shifts.freq_shift_peak_distance_ghz),
    )


def fit_scan(scan: AxialScan, sf: SpectrumFitter) -> list[AnalyzedSpectrum]:
    """fit_axial_scan(), but with a caller-supplied fitter so D stays pinned."""
    calc = CalibrationCalculator(parameters=scan.calibration_params)
    ss = scan.system_state
    cam = ss.andor_camera_info
    ratio = na_mean_shift_ratio(sf.sample_config)

    out = []
    for measurement in scan.measurements:
        frame = measurement.frame_andor.copy()
        frame = subtract_darknoise(frame=frame, darknoise_frame=ss.dark_image)

        px, sline = sf.get_px_sline_from_image(frame)
        fit = sf.fit(px=px, sline=sline, is_reference_mode=ss.is_reference_mode)
        photons = PixelCountsAndPhotons.from_fit(
            fs=fit, preamp_gain=cam.preamp_gain, emccd_gain=cam.gain)
        theo = theoretical_precision(
            fs=fit, photons=photons, calibration_calculator=calc,
            dark_frame_std=(ss.dark_image.std_image
                            if ss.dark_image is not None else None),
            preamp_gain=cam.preamp_gain, emccd_gain=cam.gain)
        out.append(AnalyzedSpectrum(fitted_spectrum=fit,
                                    analyzed_shifts=na_corrected(calc.analyze(fit), ratio),
                                    photons=photons, theoretical_precisions=theo))
    return out


def extract(scan: AxialScan):
    """Return (depth_um, shift_ghz, err_ghz, n_good, n_total) for one scan."""
    spectra = fit_scan(scan, session_fitter())
    z = np.array([m.lens_zaber_position for m in scan.measurements], float)

    def col(getter):
        return np.array([getter(s) if s.fitted_spectrum.is_success else np.nan
                         for s in spectra], float)

    # freq_shift_peak_distance_ghz is already the shift in GHz -- the calibration
    # polynomial maps inter-peak pixel distance onto the EOM frequency. Do not halve.
    shift = col(lambda s: s.analyzed_shifts.freq_shift_peak_distance_ghz)
    left = col(lambda s: s.analyzed_shifts.freq_shift_left_peak_ghz)
    right = col(lambda s: s.analyzed_shifts.freq_shift_right_peak_ghz)
    err = col(lambda s: s.theoretical_precisions.distance_total_mhz) / 1000.0

    good = (np.isfinite(shift) & np.isfinite(left) & np.isfinite(right)
            & (np.abs(left - right) < MAX_LR_DISAGREE)
            & (shift > PLAUSIBLE[0]) & (shift < PLAUSIBLE[1]))
    if not good.any():
        return None

    # Depth 0 = first valid measurement of this scan (our definition of the front).
    z0 = z[np.argmax(good)]
    return z[good] - z0, shift[good], err[good], int(good.sum()), len(z)


def plot_person(name: str, filename: str):
    scans = [s for s in load_scans(DATA / filename)
             if len(s.measurements) >= MIN_DEPTH_SCAN_POINTS]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    print(f"\n=== {name} ({filename}): {len(scans)} axial scan(s) ===")
    n_plotted = 0
    for scan in scans:
        res = extract(scan)
        step = np.median(np.diff([m.lens_zaber_position for m in scan.measurements]))
        if res is None:
            print(f"  scan i={scan.i} ({scan.id}): NO valid fits - skipped")
            continue
        depth, shift, err, n_good, n_tot = res
        ax.errorbar(depth, shift, yerr=err, lw=1.5, capsize=2.5, alpha=0.9, ms=6,
                    color=PALETTE[n_plotted % len(PALETTE)],
                    marker=MARKERS[n_plotted % len(MARKERS)],
                    label=f"scan {scan.i}  ({n_good}/{n_tot} pts, {step:.0f} µm step)")
        n_plotted += 1
        print(f"  scan i={scan.i} ({scan.id}): {n_good}/{n_tot} valid, "
              f"{shift.min():.3f}-{shift.max():.3f} GHz over {depth.max():.0f} µm")

    ax.set_xlabel("Depth from front surface — Zaber lens travel (µm)")
    ax.set_ylabel("Brillouin shift (GHz)")
    ax.set_title(f"{name} — cornea axial scans, 2026-07-23\n"
                 f"shift from inter-peak distance; depth 0 = first valid fit of each scan",
                 fontsize=10)
    # State the NA-model parameters on the figure: D is a single session-wide
    # value, not refit per point, and it is not water-calibrated for this date.
    ax.text(0.995, 0.02,
            f"{FITTING_MODEL}, D = {SESSION_D_MM} mm (house default — no 7-23 water "
            f"bracket), n = {N_SAMPLE}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.45")
    ax.grid(alpha=0.3)
    if n_plotted:
        ax.legend(fontsize=8, title=f"{n_plotted} scan(s)", title_fontsize=8)
    else:
        ax.text(0.5, 0.5, "no valid fits in any scan", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="0.4")

    secax = ax.secondary_xaxis("top", functions=(lambda d: d * N_CORNEA,
                                                lambda d: d / N_CORNEA))
    secax.set_xlabel(f"approx. focus depth in tissue (µm, ×n={N_CORNEA})", fontsize=9)

    fig.tight_layout()
    out = OUT / f"axial_depth_{name.lower()}_20260723.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for person, fn in PEOPLE.items():
        plot_person(person, fn)
