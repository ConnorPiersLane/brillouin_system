"""Solve the per-session D (na_beam_diameter_mm) from a two-NA water bracket.

Method (as in docs/na_project_status.md): the same water sits under two objectives
with different collection NA. The NA correction must return the SAME true 180 deg
shift for both. D is the only free parameter, so solve

    f180(na042, D) - f180(na014, D) = 0

Config is overridden IN MEMORY ONLY; the project TOML is never written.

Usage:
    PYTHONPATH=src python src/simple_plotting/solve_session_D_from_water.py [water .h5]
"""
import copy
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter
from brillouin_system.spectrum_fitting.na_lineshape import na_mean_shift_ratio

P = Path(sys.argv[1] if len(sys.argv) > 1 else
         r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2026-7-27\water_na014_na042.h5")
N_WATER = 1.33
# The NA lineshape models are gone (2026-08-20): this now uses the production
# post-hoc route â€” fit a plain windowed Lorentzian, then DIVIDE the shift by
# the scalar <cos(v/2)> (na_mean_shift_ratio) with the Gaussian coupling
# weight. Validated equivalent to the in-fit NA model on water.
MODEL = "lorentzian_window"
# The calibration must be fitted in the same lineshape family as the samples, and
# the live TOML tracks whatever the GUI was last set to (pixel_response/prm1 as of
# 2026-08). Pin it: mixing families moves the solved D by ~0.4 mm.
REFERENCE_MODEL = "lorentzian"
# id -> (nominal objective NA, objective focal length mm)
GEOM = {"na014": (0.14, 40.0), "na042": (0.42, 10.0)}


def pin_reference(sf: SpectrumFitter):
    """Fit the calibration with REFERENCE_MODEL, not the live TOML's model."""
    ref = copy.deepcopy(sf.reference_config)
    ref.fitting_model = REFERENCE_MODEL
    sf.update_reference_config(ref)


def load_scans(p):
    o = dict_to_dataclass_tree(load_dict_from_hdf5(str(p)), known_classes)
    return o if isinstance(o, list) else [o]


SCANS = load_scans(P)
# Cache the sline of every frame once -- it does not depend on D.
CACHE = []
for s in SCANS:
    ss = s.system_state
    sf0 = SpectrumFitter()
    slines = []
    for m in s.measurements:
        f = m.frame_andor.copy()
        slines.append(sf0.get_px_sline_from_image(f))
    CACHE.append((s, slines))
print(f"cached {sum(len(c[1]) for c in CACHE)} frames from {len(SCANS)} scans")


def group_shift(group: str, d_mm: float):
    """Mean corrected shift (GHz) over every frame of every scan in `group`."""
    na, focal = GEOM[group]
    vals = []
    for scan, slines in CACHE:
        if scan.id != group:
            continue
        sf = SpectrumFitter()
        cfg = copy.deepcopy(sf.sample_config)
        cfg.fitting_model = MODEL
        cfg.na_collection = na
        cfg.na_focal_length_mm = focal
        cfg.na_beam_diameter_mm = d_mm
        cfg.na_n_sample = N_WATER
        sf.update_sample_config(cfg)
        pin_reference(sf)

        calc = CalibrationCalculator(parameters=scan.calibration_params)
        cfg.na_weighting = "uniform_gaussian"
        ratio = na_mean_shift_ratio(cfg)

        for px, sline in slines:
            fit = sf.fit(px=px, sline=sline, is_reference_mode=False)
            if not fit.is_success:
                continue
            v = calc.analyze(fit).freq_shift_peak_distance_ghz
            if v is not None and 4.0 < v < 6.5:
                vals.append(v / ratio)
    a = np.array(vals, float)
    return a.mean(), a.std(ddof=1), a.size


def gap(d_mm):
    m42, _, _ = group_shift("na042", d_mm)
    m14, _, _ = group_shift("na014", d_mm)
    return m42 - m14


print("\n--- uncorrected (plain lorentzian_window, no NA model) ---")
for g in ["na042", "na014"]:
    sf = SpectrumFitter()
    cfg = copy.deepcopy(sf.sample_config)
    cfg.fitting_model = "lorentzian_window"
    sf.update_sample_config(cfg)
    pin_reference(sf)
    vals = []
    for scan, slines in CACHE:
        if scan.id != g:
            continue
        calc = CalibrationCalculator(parameters=scan.calibration_params)
        for px, sline in slines:
            fit = sf.fit(px=px, sline=sline, is_reference_mode=False)
            if fit.is_success:
                v = calc.analyze(fit).freq_shift_peak_distance_ghz
                if v is not None and 4.0 < v < 6.5:
                    vals.append(v)
    a = np.array(vals, float)
    print(f"  {g}: {a.mean():.4f} +/- {a.std(ddof=1)*1000:.1f} MHz (n={a.size})")

print("\n--- D scan: f180(na042) - f180(na014) ---")
print(f"{'D [mm]':>8}{'na042':>10}{'na014':>10}{'gap [MHz]':>12}")
grid = [3.5, 4.5, 5.5, 6.5, 7.5, 9.0]
gaps = []
for d in grid:
    m42, s42, n42 = group_shift("na042", d)
    m14, s14, n14 = group_shift("na014", d)
    gaps.append(m42 - m14)
    print(f"{d:8.2f}{m42:10.4f}{m14:10.4f}{(m42 - m14) * 1000:12.2f}")

sign = np.sign(gaps)
brackets = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1) if sign[i] * sign[i + 1] < 0]
if not brackets:
    print("\nNo sign change on the grid -- no self-consistent D in this range.")
else:
    lo, hi = brackets[0]
    D = brentq(gap, lo, hi, xtol=1e-3)
    m42, s42, n42 = group_shift("na042", D)
    m14, s14, n14 = group_shift("na014", D)
    print(f"\n=== SOLVED D = {D:.3f} mm  (bracket {lo}-{hi}) ===")
    print(f"  na042 f180 = {m42:.4f} +/- {s42*1000:.1f} MHz (n={n42})")
    print(f"  na014 f180 = {m14:.4f} +/- {s14*1000:.1f} MHz (n={n14})")
    print(f"  residual gap = {(m42 - m14)*1000:.3f} MHz")
    print(f"  water f180  = {(m42 + m14) / 2:.4f} GHz")
    # local sensitivity of the solve
    for dd in (D - 0.25, D + 0.25):
        print(f"  gap at D={dd:.2f}: {gap(dd)*1000:+.2f} MHz")
