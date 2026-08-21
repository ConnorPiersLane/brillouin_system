"""Are the Stokes and anti-Stokes photon numbers independent frame to frame?

Pearson r on  <(Ns - <Ns>)(Nas - <Nas>)>, deviations taken WITHIN each scan
(so scan-to-scan mean differences across the temperature series cannot
masquerade as correlation), then pooled over all scans of the 2026-8-5
NA0.14 water series.

N per peak (option 3): background-subtracted window sum. Windows are frozen
per scan from the scan-mean sline -- center +- round(beta*FWHM) px with the
production beta = 3.0 -- so window jitter never enters N. Background is the
median out-of-window level (exclusion zone 2*beta*FWHM around each peak),
scaled by the window length.

Left peak = anti-Stokes, right = Stokes (2026-08-01 Rb-lock result).

Outputs: pooled r (mean-centered and consecutive-difference variants),
per-scan r distribution, and a var/mean shot-noise check in electrons
(gain 3.5 e-/count).

Usage:
    PYTHONPATH=src python src/simple_plotting/stokes_antistokes_photon_correlation.py [folder] [glob]
    (defaults: the 2026-8-5 folder, "water_*.h5" = the NA0.14 series)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths

from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

DATA = Path(r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2026-8-5")
BETA = 3.0
GAIN_E_PER_COUNT = 3.5


def load_scans(p):
    o = dict_to_dataclass_tree(load_dict_from_hdf5(str(p)), known_classes)
    return o if isinstance(o, list) else [o]


def scan_slines(scan):
    """Background-subtracted sline of every frame, as in solve_session_D_from_water."""
    ss = scan.system_state
    sf = SpectrumFitter()
    out = []
    for m in scan.measurements:
        f = m.frame_andor.copy()
        out.append(sf.get_px_sline_from_image(f))
    return out


def peak_windows(mean_sline):
    """Two windows (anti-Stokes, Stokes) frozen from the scan-mean sline.

    Returns (windows, bg_mask) or None if two clean peaks are not found.
    windows = [(lo, hi) slice bounds), ...] ordered left -> right.
    """
    peaks, props = find_peaks(mean_sline,
                              prominence=0.001 * float(np.max(mean_sline)),
                              height=5, width=1, wlen=10)
    if len(peaks) < 2:
        return None
    top = peaks[np.argsort(props["prominences"])[-2:]]
    top = np.sort(top)
    fwhm = peak_widths(mean_sline, top, rel_height=0.5)[0]

    n = len(mean_sline)
    windows, bg_mask = [], np.ones(n, dtype=bool)
    for c, w in zip(top, fwhm):
        half = max(int(round(BETA * w)), 1)
        lo, hi = c - half, c + half + 1
        if lo < 0 or hi > n:
            return None  # window clipped by the sline edge -> N would be biased
        windows.append((lo, hi))
        ex = max(int(round(2 * BETA * w)), 1)  # background exclusion zone
        bg_mask[max(c - ex, 0):min(c + ex + 1, n)] = False
    if windows[0][1] > windows[1][0]:
        return None  # overlapping windows
    if bg_mask.sum() < 10:
        return None
    return windows, bg_mask


def photon_numbers(slines, windows, bg_mask):
    """Per frame, in electrons: (Nas, Ns) background-subtracted and raw.

    Background = median sline level outside the exclusion zones, scaled by
    the window length. The raw variant skips that step (window sum as-is).
    """
    nas, ns, nas_raw, ns_raw = [], [], [], []
    for _, sline in slines:
        bg = float(np.median(sline[bg_mask]))
        (l0, h0), (l1, h1) = windows
        s0 = float(np.sum(sline[l0:h0]))
        s1 = float(np.sum(sline[l1:h1]))
        nas.append((s0 - bg * (h0 - l0)) * GAIN_E_PER_COUNT)
        ns.append((s1 - bg * (h1 - l1)) * GAIN_E_PER_COUNT)
        nas_raw.append(s0 * GAIN_E_PER_COUNT)
        ns_raw.append(s1 * GAIN_E_PER_COUNT)
    return np.array(nas), np.array(ns), np.array(nas_raw), np.array(ns_raw)


def pooled_r(pairs):
    """Pearson r from per-scan-centered deviations pooled over scans."""
    sxy = sxx = syy = 0.0
    for x, y in pairs:
        dx, dy = x - x.mean(), y - y.mean()
        sxy += np.sum(dx * dy)
        sxx += np.sum(dx * dx)
        syy += np.sum(dy * dy)
    return sxy / np.sqrt(sxx * syy)


folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA
pattern = sys.argv[2] if len(sys.argv) > 2 else "water_*.h5"
files = sorted(folder.glob(pattern))
per_scan = []          # (file, scan idx, nas, ns)
skipped = 0
for f in files:
    for i, scan in enumerate(load_scans(f)):
        slines = scan_slines(scan)
        if len(slines) < 10:
            skipped += 1
            continue
        mean_sline = np.mean([s for _, s in slines], axis=0)
        pw = peak_windows(mean_sline)
        if pw is None:
            skipped += 1
            continue
        nas, ns, nas_raw, ns_raw = photon_numbers(slines, *pw)
        per_scan.append((f.stem, i, nas, ns, nas_raw, ns_raw))

n_frames = sum(len(nas) for _, _, nas, _, _, _ in per_scan)
n_scans = len(per_scan)
print(f"{n_scans} scans, {n_frames} frames ({skipped} scans skipped)")

# --- headline: pooled Pearson r, bg-subtracted vs raw ----------------------
pairs = [(nas, ns) for _, _, nas, ns, _, _ in per_scan]
pairs_raw = [(nas_raw, ns_raw) for _, _, _, _, nas_raw, ns_raw in per_scan]
dof = n_frames - n_scans
se = 1.0 / np.sqrt(dof)
n_d = dof
# consecutive differences share a frame -> lag-1 dependence, se ~ sqrt(1.5/n)
se_d = np.sqrt(1.5 / n_d)
for label, pp in (("bg-subtracted", pairs), ("raw (no bg sub)", pairs_raw)):
    r = pooled_r(pp)
    r_d = pooled_r([(np.diff(x), np.diff(y)) for x, y in pp])
    print(f"\n[{label}]")
    print(f"  pooled Pearson r (per-scan centered): {r:+.4f} +/- {se:.4f}  (dof={dof})")
    print(f"  pooled Pearson r (consecutive diffs):  {r_d:+.4f} +/- {se_d:.4f}  (n={n_d})")

# --- per-scan r distribution ------------------------------------------------
rs = np.array([np.corrcoef(nas, ns)[0, 1] for nas, ns in pairs])
print(f"per-scan r: mean {rs.mean():+.4f} +/- {rs.std(ddof=1) / np.sqrt(n_scans):.4f} "
      f"(sd {rs.std(ddof=1):.4f}, n={n_scans})")

# --- decomposition: common-mode size + partition (ratio) test ---------------
# If N_i = <N_i>(1 + eps) + shot with a shared relative fluctuation eps, then
# cov(Nas, Ns)/(<Nas><Ns>) = var(eps). Diff-based (x2 for the difference).
num = den = 0.0
for nas, ns in pairs:
    num += 0.5 * np.sum(np.diff(nas) * np.diff(ns)) / (nas.mean() * ns.mean())
    den += len(nas) - 1
sig_eps = np.sqrt(num / den)
print(f"common-mode relative fluctuation: {sig_eps * 100:.2f}% rms per frame "
      f"(shot would be {1/np.sqrt(np.mean([ns.mean() for _, ns in pairs])) * 100:.2f}%)")

# Partition test: p = Ns/(Ns+Nas). A common gain cancels in p; pure shot noise
# partitions the total binomially, var(p) = p(1-p)/(Nas+Ns) per frame.
exc = []
for nas, ns in pairs:
    t = nas + ns
    p = ns / t
    var_p = 0.5 * np.var(np.diff(p), ddof=1)
    var_shot = float(np.mean(p * (1 - p) / t))
    exc.append(var_p / var_shot)
exc = np.array(exc)
print(f"ratio Ns/(Ns+Nas) var vs binomial-shot prediction: "
      f"{exc.mean():.2f} +/- {exc.std(ddof=1)/np.sqrt(len(exc)):.2f}")

# --- shot-noise check: var/mean in electrons (diff-based, drift-robust) -----
print(f"\n{'scan':>16} {'<Nas>/1e3':>10} {'<Ns>/1e3':>9} {'var/mean AS':>12} {'var/mean S':>11} {'r':>7} {'r_diff':>7}")
for (name, i, nas, ns, _, _), rr in zip(per_scan, rs):
    va = 0.5 * np.var(np.diff(nas), ddof=1) / nas.mean()
    vs = 0.5 * np.var(np.diff(ns), ddof=1) / ns.mean()
    rd = np.corrcoef(np.diff(nas), np.diff(ns))[0, 1]
    print(f"{name:>14}:{i:<2d} {nas.mean() / 1e3:10.1f} {ns.mean() / 1e3:9.1f} "
          f"{va:12.2f} {vs:11.2f} {rr:+7.3f} {rd:+7.3f}")
