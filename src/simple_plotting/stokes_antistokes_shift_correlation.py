"""Are the per-peak Brillouin SHIFTS (nu_as, nu_s) frame-to-frame independent?

Companion to stokes_antistokes_photon_correlation.py (which does the photon
NUMBERS): pooled Pearson r on <(nu_s - <nu_s>)(nu_as - <nu_as>)>, deviations
centered WITHIN each scan, pooled over the 2026-8-5 NA0.14 water series.

Per frame: production prm1 fit (default TOML config), each scan's own
calibration (per-scan calibration rule), nu_as = freq_shift_left_peak_ghz,
nu_s = freq_shift_right_peak_ghz.

Also decomposes into common mode (nu_s + nu_as)/2 vs differential
(nu_s - nu_as)/2 -- under independence both have the same variance; the
distance gains sqrt(2) only if r ~ 0.

Usage:
    PYTHONPATH=src python src/simple_plotting/stokes_antistokes_shift_correlation.py [folder] [glob]
    (defaults: the 2026-8-5 folder, "water_*.h5" = the NA0.14 series)
"""
import sys
from pathlib import Path

import numpy as np

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree
from brillouin_system.spectrum_fitting.helpers.subtract_darknoise import subtract_darknoise
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

DATA = Path(r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2026-8-5")


def load_scans(p):
    o = dict_to_dataclass_tree(load_dict_from_hdf5(str(p)), known_classes)
    return o if isinstance(o, list) else [o]


def scan_shifts(scan):
    """(nu_as, nu_s) per frame in GHz from the production fit."""
    ss = scan.system_state
    sf = SpectrumFitter()
    calc = CalibrationCalculator(parameters=scan.calibration_params)
    vas, vs = [], []
    for m in scan.measurements:
        f = m.frame_andor.copy()
        f = subtract_darknoise(frame=f, darknoise_frame=ss.dark_image)
        px, sline = sf.get_px_sline_from_image(f)
        fit = sf.fit(px=px, sline=sline, is_reference_mode=False)
        if not fit.is_success:
            continue
        res = calc.analyze(fit)
        l, r = res.freq_shift_left_peak_ghz, res.freq_shift_right_peak_ghz
        if l is None or r is None or not (4.0 < l < 6.5) or not (4.0 < r < 6.5):
            continue
        vas.append(l)
        vs.append(r)
    return np.array(vas), np.array(vs)


def pooled_r(pairs):
    sxy = sxx = syy = 0.0
    for x, y in pairs:
        dx, dy = x - x.mean(), y - y.mean()
        sxy += np.sum(dx * dy)
        sxx += np.sum(dx * dx)
        syy += np.sum(dy * dy)
    return sxy / np.sqrt(sxx * syy)


per_scan = []
skipped = 0
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA
pattern = sys.argv[2] if len(sys.argv) > 2 else "water_*.h5"
for f in sorted(folder.glob(pattern)):
    for i, scan in enumerate(load_scans(f)):
        vas, vs = scan_shifts(scan)
        if len(vas) < 10:
            skipped += 1
            continue
        per_scan.append((f.stem, i, vas, vs))

n_frames = sum(len(v) for _, _, v, _ in per_scan)
n_scans = len(per_scan)
print(f"{n_scans} scans, {n_frames} frames ({skipped} scans skipped)")

pairs = [(vas, vs) for _, _, vas, vs in per_scan]
dof = n_frames - n_scans
se = 1.0 / np.sqrt(dof)
r = pooled_r(pairs)
r_d = pooled_r([(np.diff(x), np.diff(y)) for x, y in pairs])
print(f"\npooled Pearson r (per-scan centered): {r:+.4f} +/- {se:.4f}  (dof={dof})")
print(f"pooled Pearson r (consecutive diffs):  {r_d:+.4f} +/- {np.sqrt(1.5 / dof):.4f}")

rs = np.array([np.corrcoef(x, y)[0, 1] for x, y in pairs])
print(f"per-scan r: mean {rs.mean():+.4f} +/- {rs.std(ddof=1) / np.sqrt(n_scans):.4f} "
      f"(sd {rs.std(ddof=1):.4f}, n={n_scans})")

# --- pooled sds + common/differential decomposition (MHz) -------------------
def pooled_sd(devs):
    return np.sqrt(sum(np.sum(d * d) for d in devs) / dof)

d_as = [x - x.mean() for x, _ in pairs]
d_s = [y - y.mean() for _, y in pairs]
sd_as = pooled_sd(d_as) * 1000
sd_s = pooled_sd(d_s) * 1000
sd_mean = pooled_sd([(a + b) / 2 for a, b in zip(d_as, d_s)]) * 1000
sd_half = pooled_sd([(b - a) / 2 for a, b in zip(d_as, d_s)]) * 1000
print(f"\npooled per-frame sd [MHz]: nu_as {sd_as:.2f}, nu_s {sd_s:.2f}")
print(f"  common (nu_s+nu_as)/2:       {sd_mean:.2f}")
print(f"  differential (nu_s-nu_as)/2: {sd_half:.2f}")
print(f"  independence predicts both = {np.sqrt(sd_as**2 + sd_s**2) / 2:.2f}")

print(f"\n{'scan':>16} {'n':>3} {'sd_as':>6} {'sd_s':>6} {'r':>7} {'r_diff':>7}")
for (name, i, vas, vs), rr in zip(per_scan, rs):
    rd = np.corrcoef(np.diff(vas), np.diff(vs))[0, 1]
    print(f"{name:>14}:{i:<2d} {len(vas):3d} {vas.std(ddof=1)*1000:6.2f} "
          f"{vs.std(ddof=1)*1000:6.2f} {rr:+7.3f} {rd:+7.3f}")
