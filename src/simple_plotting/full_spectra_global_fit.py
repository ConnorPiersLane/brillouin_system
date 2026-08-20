"""Full-CCD multi-order calibration + global N-Lorentzian fit (PI request 2026-08-04).

Data: "full_spectra" files from 2026-07-31 (full-chip ROI, 243 px wide).

Step A - calibration: track every etalon-order EOM sideband across the 41
    calibration frequencies (4.0-8.0 GHz); fit each consecutive pair distance
    d_k(f) with a quadratic (and its inverse f(d_k), same convention as the
    production freq_peak_distance polynomial).
Step B - fit each water frame two ways:
    conventional: two independent Lorentzians on the main pair, distance -> shift
    global: sum of 8 Lorentzians whose positions are ALL tied to one shift
        variable nu through the d_k(f) curves, plus one global offset x0.
        Free per peak: amplitude and width. Stray peaks between the main pair
        (px 106-119) are masked.
Step C - per-pair shift estimates from individual peak fits, and the
    inverse-variance combination, as a cross-check of the global-fit gain.

Result on this data: pooled per-frame sd 5.0 MHz (conventional) vs 4.4 MHz
(global) -> x1.15 precision gain; inverse-variance combo of all 7 pair
distances gives 4.0 MHz (x1.23 bound).

Usage:
    PYTHONPATH=src python src/simple_plotting/full_spectra_global_fit.py [out_dir]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks

DATA = Path(r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2026-7-31")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

ROWS = slice(8, 21)      # rows of the 26-row chip that carry signal
STRAY = (106, 119)       # stray peaks between the main pair: masked in global fit
MAIN_PAIR = 2            # d[2] = distance between px~97 and px~124


def frame_profile(frame):
    p = frame[ROWS].astype(float).mean(axis=0)
    return p - np.median(p)


def lorentz(x, a, x0, w, c):
    return a * w**2 / ((x - x0) ** 2 + w**2) + c


def fit_single_peak(prof, i0, half=7):
    lo, hi = max(0, int(round(i0)) - half), min(len(prof), int(round(i0)) + half + 1)
    x = np.arange(lo, hi, dtype=float)
    y = prof[lo:hi]
    p0 = [max(y.max() - y.min(), 1e-3), float(i0), 2.0, float(np.min(y))]
    popt, _ = curve_fit(lorentz, x, y, p0=p0,
                        bounds=([0, lo, 0.5, -np.inf], [np.inf, hi, 8, np.inf]),
                        maxfev=8000)
    return popt


# ====================================================================== Step A
def build_calibration():
    with open(DATA / "full_spectra_calibration.pkl", "rb") as f:
        cal = pickle.load(f)
    entries = sorted(cal.measured_freqs, key=lambda m: m.set_freq_ghz)
    freqs = np.array([m.set_freq_ghz for m in entries])

    pos_per_freq = []
    for m in entries:
        frames = [pt.frame for pt in m.cali_meas_points]
        prof = frame_profile(np.stack(frames).astype(float).mean(axis=0))
        idx, _ = find_peaks(prof, prominence=max(2.0, 0.03 * prof.max()), distance=6)
        fitted = []
        for i in idx:
            try:
                r = fit_single_peak(prof, i, half=6)
            except RuntimeError:
                continue
            if r[0] > 2.0:
                fitted.append(r[1])
        pos_per_freq.append(np.array(sorted(fitted)))

    # track identities from the frequency nearest 5.0 GHz (water sits there)
    i_anchor = int(np.argmin(np.abs(freqs - 5.0)))
    anchor = pos_per_freq[i_anchor]
    n = len(anchor)
    tracks = np.full((len(freqs), n), np.nan)
    tracks[i_anchor] = anchor

    def match(prev, new, tol=4.0):
        out = np.full(len(prev), np.nan)
        for k, p in enumerate(prev):
            if np.isnan(p):
                continue
            d = np.abs(new - p)
            j = int(np.argmin(d))
            if d[j] < tol:
                out[k] = new[j]
        return out

    for i in range(i_anchor + 1, len(freqs)):
        tracks[i] = match(tracks[i - 1], pos_per_freq[i])
    for i in range(i_anchor - 1, -1, -1):
        tracks[i] = match(tracks[i + 1], pos_per_freq[i])

    dist = np.diff(tracks, axis=1)
    coef_d_of_f, coef_f_of_d = [], []
    for k in range(n - 1):
        m = ~np.isnan(dist[:, k])
        coef_d_of_f.append(np.polyfit(freqs[m], dist[m, k], 2))
        coef_f_of_d.append(np.polyfit(dist[m, k], freqs[m], 2))
        r = dist[m, k] - np.polyval(coef_d_of_f[k], freqs[m])
        print(f"d[{k}]: span {dist[m, k].min():6.2f}-{dist[m, k].max():6.2f} px, "
              f"quad resid rms {r.std() * 1000:5.1f} mpx")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k in range(n):
        axes[0].plot(freqs, tracks[:, k], ".-", ms=3, lw=0.7)
    axes[0].set(xlabel="EOM frequency (GHz)", ylabel="peak position (px)",
                title="Tracked sideband positions, all orders")
    ff = np.linspace(4, 8, 200)
    for k in range(n - 1):
        m = ~np.isnan(dist[:, k])
        l, = axes[1].plot(freqs[m], dist[m, k], ".", ms=4)
        axes[1].plot(ff, np.polyval(coef_d_of_f[k], ff), "-", lw=0.8,
                     color=l.get_color(), label=f"d{k}")
    axes[1].set(xlabel="EOM frequency (GHz)", ylabel="consecutive peak distance (px)",
                title="Pairwise distances + quadratic fits")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_tracks.png", dpi=140)
    return np.array(coef_d_of_f), np.array(coef_f_of_d), anchor


# ====================================================================== Step B
def run_water(coef_d_of_f, coef_f_of_d, anchor):
    n_peaks = len(anchor)

    def rel_positions(nu):
        d = np.array([np.polyval(c, nu) for c in coef_d_of_f])
        return np.concatenate([[0.0], np.cumsum(d)])

    with open(DATA / "full_spectra_water.pkl", "rb") as f:
        water = pickle.load(f)
    profiles, scan_ids = [], []
    for s in water:
        for m in s.measurements:
            profiles.append(frame_profile(m.frame_andor))
            scan_ids.append(s.i)
    profiles = np.array(profiles)
    scan_ids = np.array(scan_ids)
    nx = profiles.shape[1]
    xgrid = np.arange(nx, dtype=float)
    mask = np.ones(nx, dtype=bool)
    mask[STRAY[0]:STRAY[1]] = False

    # conventional: two independent Lorentzians on the main pair
    nu_conv = np.full(len(profiles), np.nan)
    for i, prof in enumerate(profiles):
        try:
            pl = fit_single_peak(prof, anchor[MAIN_PAIR])
            pr = fit_single_peak(prof, anchor[MAIN_PAIR + 1])
            nu_conv[i] = np.polyval(coef_f_of_d[MAIN_PAIR], pr[1] - pl[1])
        except RuntimeError:
            pass

    # global: all peaks tied to a single shift variable
    def global_model(params, x):
        nu, x0 = params[0], params[1]
        amps = params[2:2 + n_peaks]
        ws = params[2 + n_peaks:2 + 2 * n_peaks]
        bg = params[-1]
        pos = x0 + rel_positions(nu)
        y = np.full_like(x, bg)
        for a, p, w in zip(amps, pos, ws):
            y = y + a * w**2 / ((x - p) ** 2 + w**2)
        return y

    def fit_global(prof, nu0, x00):
        pos0 = x00 + rel_positions(nu0)
        a0 = np.clip(prof[np.clip(np.round(pos0).astype(int), 0, nx - 1)], 0.5, None)
        p0 = np.concatenate([[nu0, x00], a0, np.full(n_peaks, 2.0), [0.0]])
        lo = np.concatenate([[4.2, x00 - 6], np.zeros(n_peaks),
                             np.full(n_peaks, 0.8), [-5]])
        hi = np.concatenate([[7.8, x00 + 6], np.full(n_peaks, np.inf),
                             np.full(n_peaks, 8.0), [5]])

        def resid(p):
            return (global_model(p, xgrid) - prof)[mask]

        sol = least_squares(resid, p0, bounds=(lo, hi), method="trf", max_nfev=4000)
        return sol

    nu_glob = np.full(len(profiles), np.nan)
    x0_seed = anchor[0]
    for i, prof in enumerate(profiles):
        nu0 = nu_conv[i] if np.isfinite(nu_conv[i]) else 5.0
        try:
            sol = fit_global(prof, nu0, x0_seed)
            nu_glob[i] = sol.x[0]
            x0_seed = sol.x[1]
        except Exception as e:
            print("global fit failed on frame", i, e)

    # Step C: per-pair estimates from individual peak fits
    pos = np.full((len(profiles), n_peaks), np.nan)
    for i, prof in enumerate(profiles):
        for j, a in enumerate(anchor):
            try:
                pos[i, j] = fit_single_peak(prof, a)[1]
            except RuntimeError:
                pass

    def pooled_dev(v):
        out = [v[scan_ids == s] - np.nanmean(v[scan_ids == s])
               for s in np.unique(scan_ids)]
        out = np.concatenate(out)
        return out[np.isfinite(out)]

    sd_pairs, nu_pairs = [], []
    for k in range(n_peaks - 1):
        nu_k = np.polyval(coef_f_of_d[k], pos[:, k + 1] - pos[:, k])
        nu_pairs.append(nu_k)
        sd_pairs.append(pooled_dev(nu_k).std(ddof=1))
        print(f"pair d[{k}] ({anchor[k]:.0f}-{anchor[k + 1]:.0f} px): "
              f"sd {sd_pairs[-1] * 1000:6.2f} MHz")
    w = 1 / np.array(sd_pairs) ** 2
    nu_comb = np.nansum(np.array(nu_pairs).T * w, axis=1) / w.sum()
    print(f"inverse-variance combo of all pairs: sd "
          f"{pooled_dev(nu_comb).std(ddof=1) * 1000:.2f} MHz")

    # report
    for sid in np.unique(scan_ids):
        m = scan_ids == sid
        c = nu_conv[m][np.isfinite(nu_conv[m])]
        g = nu_glob[m][np.isfinite(nu_glob[m])]
        print(f"scan {sid}: conv {c.mean() * 1000:7.1f} +- {c.std(ddof=1) * 1000:5.2f} MHz | "
              f"glob {g.mean() * 1000:7.1f} +- {g.std(ddof=1) * 1000:5.2f} MHz | "
              f"gain x{c.std(ddof=1) / g.std(ddof=1):.2f}")
    sc = pooled_dev(nu_conv).std(ddof=3)
    sg = pooled_dev(nu_glob).std(ddof=3)
    print(f"pooled per-frame sd: conventional {sc * 1000:.2f} MHz, "
          f"global {sg * 1000:.2f} MHz -> gain x{sc / sg:.3f}")

    # figure
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    prof = profiles[0]
    sol = fit_global(prof, nu_conv[0], anchor[0])
    axes[0].plot(xgrid, prof, "k.", ms=3, label="water frame 0")
    axes[0].plot(xgrid, global_model(sol.x, xgrid), "r-", lw=1,
                 label=f"global 8-peak fit, shift = {sol.x[0] * 1000:.0f} MHz")
    axes[0].axvspan(*STRAY, color="0.85", label="masked (stray)")
    axes[0].set(xlabel="pixel", ylabel="counts",
                title="Global fit example (single 0.1 s frame)")
    axes[0].legend(fontsize=8)

    for sid, mk in zip(np.unique(scan_ids), "o^s"):
        m = scan_ids == sid
        axes[1].plot(np.where(m)[0], (nu_conv[m] - np.nanmean(nu_conv[m])) * 1000,
                     "b" + mk, ms=3, alpha=0.5)
        axes[1].plot(np.where(m)[0], (nu_glob[m] - np.nanmean(nu_glob[m])) * 1000,
                     "r" + mk, ms=3, alpha=0.5)
    axes[1].set(xlabel="frame", ylabel="shift - scan mean (MHz)",
                title=f"conventional (blue) sd={sc * 1000:.1f} MHz, "
                      f"global (red) sd={sg * 1000:.1f} MHz -> gain x{sc / sg:.2f}")
    fig.tight_layout()
    fig.savefig(OUT / "global_fit_water.png", dpi=140)
    print("saved", OUT / "global_fit_water.png")


if __name__ == "__main__":
    coef_d_of_f, coef_f_of_d, anchor = build_calibration()
    run_water(coef_d_of_f, coef_f_of_d, anchor)
