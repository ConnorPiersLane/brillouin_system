"""Cornea Brillouin shift vs. depth from the 2026-08-21 SWEEP scans.

These are true in-out sweep scans (AxialScan.sweep_cycles): every cycle finds
the surface on the way IN, takes ONE frame at target_depth past it, and finds
the surface again on the way OUT. The pair is the depth-registration check --
the same gate as the 2026-07-30 analysis, but now available per FRAME instead
of per scan:

    gate:  both crossings found AND |in - out| <= 50 um
    depth: lens_z - (in + out)/2      (pair average cancels the latency bias)

Unlike the 7-30 files these scans carry their RAW calibration frames, so each
scan's calibration is RE-FITTED from its own frames in the same lineshape
family as the samples (calibration_calculator_for_scan). That removes the
model-mixing constraint that forced plain Lorentzian fits on 7-30 and lets the
production recipe run: prm1 = lorentzian_x_psf + per-peak linear background,
windowed at beta = 3.

NA: the high-NA correction is POST-HOC (the NA lineshape models were removed
2026-08-20) -- fit as at low NA, then divide the shifts by <cos(v/2)>.
NA 0.42 with the Gaussian coupling weight and D = 6.2 mm (the user's stated
value; this session has no water bracket of its own), n_sample = 1.376.
The correction is +15.5 MHz on a 5.68 GHz shift and is COMMON-MODE: it slides
every point together and cannot create or remove a depth trend.

Usage:
    PYTHONPATH=src python src/simple_plotting/plot_cornea_sweep_20260821.py <out_dir>
"""
import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brillouin_system.calibration.calibration import calibration_calculator_for_scan
from brillouin_system.my_dataclasses.axial_scan import AxialScan
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    load_dict_from_hdf5, dict_to_dataclass_tree,
)
from brillouin_system.spectrum_fitting.na_lineshape import na_mean_shift_ratio
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

DATA = Path(r"C:\Users\cplan\Dropbox (Personal)\Boston\Data\2026-8-21")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

N_CORNEA = 1.376
PLAUSIBLE = (4.0, 8.0)      # GHz, physically possible cornea Brillouin shift
MAX_LR_DISAGREE = 0.3       # GHz; left/right must agree or the fit caught noise
MAX_PLANE_GAP_UM = 50.0     # the strict depth-registration gate

# Production recipe. The calibration is re-fitted per scan from its own raw
# frames in the SAME family (lorentzian_x_psf), so no model mixing.
FITTING_MODEL = "prm1"
REFERENCE_MODEL = "lorentzian_x_psf"
# The main pair only, so these numbers are directly comparable with 7-30. The
# ROI does hold all four VIPA orders (peaks ~36/83/116/146), so n_peaks = 4 is
# available -- it buys precision, not absolute accuracy (outer-order medians
# spread ~14 MHz), so it is deliberately not the default here.
N_PEAKS = 2

# No water bracket in this session; D stated by the user.
SESSION_D_MM = 6.2
N_SAMPLE = 1.376
NA_COLLECTION = 0.42
NA_FOCAL_LENGTH_MM = 10.0

PEOPLE = {"Connor": "connor.h5", "Jimmy": "jimmy.h5", "Zuriel": "zuriel.h5"}


def load_scans(path: Path) -> list[AxialScan]:
    obj = dict_to_dataclass_tree(load_dict_from_hdf5(str(path)), known_classes)
    return obj if isinstance(obj, list) else [obj]


def session_fitter() -> SpectrumFitter:
    """One fitter for the whole session: pinned model, NA and peak count.

    Overrides the live TOML in memory only. The row band deliberately follows
    the live sline config (the standing rule) -- calibration and samples share
    this fitter, so the band is common-mode.
    """
    sf = SpectrumFitter()

    cfg = copy.deepcopy(sf.sample_config)
    cfg.fitting_model = FITTING_MODEL
    cfg.na_weighting = "uniform_gaussian"
    cfg.na_collection = NA_COLLECTION
    cfg.na_beam_diameter_mm = SESSION_D_MM
    cfg.na_focal_length_mm = NA_FOCAL_LENGTH_MM
    cfg.na_n_sample = N_SAMPLE
    sf.update_sample_config(cfg)

    ref = copy.deepcopy(sf.reference_config)
    ref.fitting_model = REFERENCE_MODEL
    sf.update_reference_config(ref)

    sl = copy.deepcopy(sf.sline_config)
    sl.n_peaks = N_PEAKS
    sf.update_sline_config(sl)
    return sf


def na_corrected(shifts, ratio):
    """Divide the shifts by the post-hoc NA ratio (widths untouched)."""
    def d(v):
        return None if v is None else v / ratio
    return replace(
        shifts,
        freq_shift_left_peak_ghz=d(shifts.freq_shift_left_peak_ghz),
        freq_shift_right_peak_ghz=d(shifts.freq_shift_right_peak_ghz),
        freq_shift_peak_distance_ghz=d(shifts.freq_shift_peak_distance_ghz),
    )


def crossing_z(result):
    """Surface z of one crossing, or None when the finder did not fire."""
    if result is None or not getattr(result, "found", False):
        return None
    z = getattr(result, "event_z_um", None)
    return None if z is None else float(z)


def collect_scan(scan: AxialScan, sf: SpectrumFitter, ratio: float):
    """One row per sweep cycle, kept or dropped, with the reason."""
    calc = calibration_calculator_for_scan(
        scan.calibration_data, scan.calibration_params, sf)
    ss = scan.system_state
    target = (float(scan.sweep_config.target_depth_um)
              if scan.sweep_config is not None else np.nan)

    rows = []
    for cycle in (scan.sweep_cycles or []):
        z_in = crossing_z(cycle.reflection_in)
        z_out = crossing_z(cycle.reflection_out)
        gap = (z_in - z_out) if (z_in is not None and z_out is not None) else np.nan
        mi = cycle.measurement_index

        shift = left = right = depth = np.nan
        if mi is not None and 0 <= mi < len(scan.measurements):
            m = scan.measurements[mi]
            px, sline = sf.get_px_sline_from_image(m.frame_andor.copy())
            fit = sf.fit(px=px, sline=sline, is_reference_mode=ss.is_reference_mode)
            if fit.is_success:
                a = na_corrected(calc.analyze(fit), ratio)
                def num(v):
                    return np.nan if v is None else float(v)
                shift = num(a.freq_shift_peak_distance_ghz)
                left = num(a.freq_shift_left_peak_ghz)
                right = num(a.freq_shift_right_peak_ghz)
            if np.isfinite(gap):
                depth = float(m.lens_zaber_position) - 0.5 * (z_in + z_out)

        pass_plane = np.isfinite(gap) and abs(gap) <= MAX_PLANE_GAP_UM
        pass_fit = (np.isfinite(shift) and np.isfinite(left) and np.isfinite(right)
                    and abs(left - right) < MAX_LR_DISAGREE
                    and PLAUSIBLE[0] < shift < PLAUSIBLE[1])

        if mi is None:
            why = "no frame taken (missed in-crossing)"
        elif not np.isfinite(gap):
            why = "only one crossing found"
        elif not pass_plane:
            why = f"crossings disagree {gap:+.0f} um"
        elif not np.isfinite(shift):
            why = "fit failed"
        elif not (PLAUSIBLE[0] < shift < PLAUSIBLE[1]):
            why = f"shift {shift:.3f} GHz outside {PLAUSIBLE}"
        elif abs(left - right) >= MAX_LR_DISAGREE:
            why = f"L-R disagree {(left-right)*1000:+.0f} MHz"
        else:
            why = "kept"

        rows.append(dict(scan_i=int(scan.i), scan_id=str(scan.id), target=target,
                         cycle=int(cycle.cycle_index), gap=gap, depth=depth,
                         shift=shift, fit_ok=pass_fit,
                         keep=pass_plane and pass_fit, why=why))
    return rows


def collect_person(filename: str, sf: SpectrumFitter, ratio: float):
    rows = []
    for scan in load_scans(DATA / filename):
        rows += collect_scan(scan, sf, ratio)
    return rows


def summarize(name: str, rows: list[dict]):
    kept = [r for r in rows if r["keep"]]
    n_plane = sum(1 for r in rows if np.isfinite(r["gap"])
                  and abs(r["gap"]) <= MAX_PLANE_GAP_UM)
    print(f"\n=== {name}: {len(rows)} cycles -> {n_plane} pass the "
          f"{MAX_PLANE_GAP_UM:.0f} um gate -> {len(kept)} also fit")
    by_scan = {}
    for r in rows:
        by_scan.setdefault((r["scan_i"], r["scan_id"], r["target"]), []).append(r)
    for (i, sid, target), rs in sorted(by_scan.items()):
        k = [r for r in rs if r["keep"]]
        gaps = np.array([abs(r["gap"]) for r in rs if np.isfinite(r["gap"])])
        line = (f"  i={i:>3} {sid:<14} target {target:>5.0f} um: "
                f"{len(k)}/{len(rs)} kept, median |in-out| "
                f"{np.median(gaps):.0f} um" if gaps.size else
                f"  i={i:>3} {sid:<14}: no crossings")
        if len(k) > 1:
            y = np.array([r["shift"] for r in k])
            d = np.array([r["depth"] for r in k])
            line += (f", depth {d.mean():.0f} um, "
                     f"{y.mean():.4f} GHz +/- {y.std(ddof=1)*1000:.1f} MHz sd")
        elif len(k) == 1:
            line += f", single point {k[0]['shift']:.4f} GHz"
        print(line)
        for r in rs:
            if not r["keep"]:
                print(f"        drop cycle {r['cycle']:>2}: {r['why']}")
    return kept


# Cornea ~5.70 GHz, aqueous humor ~5.19 GHz (2026-07-23 axial scans). A frame
# below this threshold has left the cornea -- the posterior surface is a step,
# not a gradient, so a single midpoint classifies cleanly.
AQUEOUS_BELOW_GHZ = 5.45


def by_target(kept: list[dict]):
    """Aggregate kept cycles per scan: (depth, mean shift, sd, n, target).

    A scan whose frames straddle the posterior step gets no aggregate: its
    mean sits in a gap where no tissue exists, and its sd is the step height,
    not a measurement error. Those scans are reported as `straddles`.
    """
    out = []
    groups = {}
    for r in kept:
        groups.setdefault(r["scan_i"], []).append(r)
    for i, rs in sorted(groups.items()):
        y = np.array([r["shift"] for r in rs])
        d = np.array([r["depth"] for r in rs])
        n_aq = int((y < AQUEOUS_BELOW_GHZ).sum())
        out.append(dict(scan_i=i, target=rs[0]["target"], n=len(rs),
                        depth=d.mean(), shift=y.mean(),
                        sd=(y.std(ddof=1) if len(y) > 1 else np.nan),
                        straddles=(0 < n_aq < len(y))))
    return out


def step_location(kept: list[dict]):
    """Where the focus leaves the cornea: last all-cornea and first all-aqueous
    commanded depth, plus the 50%-crossing estimate."""
    per_target = {}
    for r in kept:
        per_target.setdefault(r["target"], []).append(r["shift"] < AQUEOUS_BELOW_GHZ)
    rows = sorted((t, np.mean(v), len(v)) for t, v in per_target.items())
    below = [t for t, f, _ in rows if f == 0]
    above = [t for t, f, _ in rows if f == 1]
    cross = next((t for t, f, _ in rows if f >= 0.5), np.nan)
    return rows, (max(below) if below else np.nan), (min(above) if above else np.nan), cross


def gate_sensitivity(all_rows: dict):
    """Does the gate change the answer, or only the sample size?

    Restricted to CORNEA frames (above the aqueous threshold): pooling across
    the posterior step would mix two tissues and the mean would just track how
    many aqueous frames each gate happened to admit.
    """
    print("\n=== plane-gate sensitivity, cornea frames only (same fits) ===")
    print(f"{'subject':>10}{'gate':>8}{'n':>5}{'mean [GHz]':>13}{'sd [MHz]':>11}{'sem':>8}")
    for name, rows in all_rows.items():
        for gate in (25.0, 50.0, 100.0, 1e9):
            sel = [r for r in rows if r["fit_ok"] and np.isfinite(r["gap"])
                   and abs(r["gap"]) <= gate
                   and r["shift"] >= AQUEOUS_BELOW_GHZ]
            if len(sel) < 2:
                continue
            y = np.array([r["shift"] for r in sel])
            lbl = f"{gate:.0f}" if gate < 1e8 else "none"
            print(f"{name:>10}{lbl:>8}{len(y):>5}{y.mean():13.4f}"
                  f"{y.std(ddof=1)*1000:11.1f}"
                  f"{y.std(ddof=1)/np.sqrt(len(y))*1000:8.1f}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sf = session_fitter()
    ratio = na_mean_shift_ratio(sf.sample_config)
    print(f"recipe: {FITTING_MODEL} samples vs {REFERENCE_MODEL} calibration "
          f"(re-fitted per scan from its own frames), n_peaks={N_PEAKS}, "
          f"rows {sf.sline_config.selected_rows[0]}-{sf.sline_config.selected_rows[-1]}")
    print(f"NA: uniform_gaussian, NA {NA_COLLECTION}, D = {SESSION_D_MM} mm, "
          f"n = {N_SAMPLE} -> ratio {ratio:.5f} (+{(5.68/ratio-5.68)*1000:.1f} MHz "
          f"on 5.68 GHz)")

    all_rows, all_kept = {}, {}
    for name, fn in PEOPLE.items():
        rows = collect_person(fn, sf, ratio)
        all_rows[name] = rows
        all_kept[name] = summarize(name, rows)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=True,
                             gridspec_kw={"width_ratios": [1, 2, 1]})
    for ax, name in zip(axes, PEOPLE):
        kept = all_kept[name]
        if kept:
            d = np.array([r["depth"] for r in kept])
            y = np.array([r["shift"] for r in kept])
            aq = y < AQUEOUS_BELOW_GHZ
            ax.plot(d[~aq], y[~aq], "o", ms=5, color="#8FB8DE",
                    markeredgecolor="none", zorder=2,
                    label=f"single cycles, cornea (n={int((~aq).sum())})")
            if aq.any():
                ax.plot(d[aq], y[aq], "o", ms=5, color="#E8A87C",
                        markeredgecolor="none", zorder=2,
                        label=f"single cycles, aqueous (n={int(aq.sum())})")

            agg = by_target(kept)
            good = [a for a in agg if not a["straddles"]]
            if good:
                ax.errorbar([a["depth"] for a in good], [a["shift"] for a in good],
                            yerr=[a["sd"] for a in good], ls="none", marker="o",
                            ms=9, capsize=3.5, lw=1.4, color="#0072B2",
                            markeredgecolor="black", markeredgewidth=0.6, zorder=4,
                            label="per-scan mean ± sd")
            if any(a["straddles"] for a in agg):
                ax.plot([], [], " ", label="(scans across the step: points only)")

            # Plateau statistics over PURE depths only: commanded depths where
            # every kept frame landed in the same tissue. Frames at a depth that
            # sometimes reads cornea and sometimes aqueous straddle the posterior
            # surface, so their scatter is the step height, not tissue variation.
            # "Pure" = every kept frame at that commanded depth read the same
            # tissue AND they agree with each other. The second condition is
            # what excludes the boundary depths: at 300-400 um single frames
            # dip to 5.49-5.57 GHz (the focal volume straddling the posterior
            # surface), which reads as a gradient if those depths are kept.
            t = np.array([r["target"] for r in kept])
            def consistent(mask_tissue):
                out = {}
                for tt in np.unique(t):
                    yy = y[t == tt]
                    same = mask_tissue(yy).all()
                    tight = yy.std(ddof=1) * 1000 <= 25.0 if yy.size > 1 else True
                    out[tt] = same and tight
                return np.array([out[tt] for tt in t])
            pure_cornea = consistent(lambda v: v >= AQUEOUS_BELOW_GHZ)
            pure_aqueous = consistent(lambda v: v < AQUEOUS_BELOW_GHZ)
            for sel, col, lbl in ((pure_cornea, "#0072B2", "cornea"),
                                  (pure_aqueous, "#C1662F", "aqueous")):
                if sel.sum() > 2:
                    m, s = y[sel].mean(), y[sel].std(ddof=1)
                    span = (f"≤{t[sel].max():.0f}" if lbl == "cornea"
                            else f"≥{t[sel].min():.0f}")
                    ax.axhspan(m - s, m + s, color=col, alpha=0.10, zorder=0,
                               label=f"{lbl} ({span} µm, n={int(sel.sum())}): "
                                     f"{m:.4f} ± {s*1000:.0f} MHz sd")
                    ax.axhline(m, color=col, lw=1.0, alpha=0.55, zorder=1)

            # Is there a gradient THROUGH the stroma, or is it a flat plateau?
            if pure_cornea.sum() > 5 and float(np.ptp(d[pure_cornea])) > 150:
                dc, yc = d[pure_cornea], y[pure_cornea]
                slope, icept = np.polyfit(dc, yc, 1)
                resid = yc - (icept + slope * dc)
                se = resid.std(ddof=2) / np.sqrt(((dc - dc.mean()) ** 2).sum())
                print(f"  {name}: stromal gradient {slope*1e6:+.1f} ± {se*1e6:.1f} "
                      f"MHz per mm of travel ({slope*1e6/N_CORNEA:+.1f} MHz/mm in "
                      f"tissue) over {dc.min():.0f}-{dc.max():.0f} um, n={len(yc)}")
                xs = np.linspace(dc.min(), dc.max(), 40)
                ax.plot(xs, icept + slope * xs, "-", color="0.30", lw=1.3, zorder=3,
                        label=f"stroma slope {slope*1e6:+.0f} ± {se*1e6:.0f} MHz/mm")
        ax.set_title(name, fontsize=11)
        # Where the focus leaves the cornea (only meaningful on a depth series).
        rows, last_cornea, first_aq, cross = step_location(kept) if kept else ([], np.nan, np.nan, np.nan)
        if len(rows) > 3 and np.isfinite(cross):
            ax.axvline(cross, color="0.4", ls="--", lw=1.1, zorder=1)
            ax.text(cross - 12, 0.60, f"posterior surface ≈ {cross:.0f} µm travel\n"
                                      f"≈ {cross*N_CORNEA:.0f} µm thickness  ",
                    transform=ax.get_xaxis_transform(), fontsize=7.5,
                    color="0.35", va="center", ha="right")
            print(f"\n  {name}: aqueous fraction per commanded depth "
                  f"{[(int(t), round(f, 2), n) for t, f, n in rows]}")
            print(f"  {name}: last all-cornea {last_cornea:.0f} um, first "
                  f"all-aqueous {first_aq:.0f} um, 50% crossing {cross:.0f} um "
                  f"travel -> {cross*N_CORNEA:.0f} um optical thickness")

        ax.set_xlabel("Depth past the surface — Zaber lens travel (µm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower left")
        secax = ax.secondary_xaxis("top", functions=(lambda d: d * N_CORNEA,
                                                     lambda d: d / N_CORNEA))
        secax.set_xlabel(f"optical depth in tissue (µm, ×n={N_CORNEA})", fontsize=8)
    axes[0].set_ylabel("Brillouin shift (GHz)")
    # Headroom under the data so the legends never sit on a point.
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo - 0.42 * (hi - lo), hi)

    fig.suptitle(
        "Cornea Brillouin shift vs. depth — 2026-08-21 sweep scans\n"
        f"one frame per in-out cycle, kept only when the two crossings agree "
        f"within {MAX_PLANE_GAP_UM:.0f} µm; depth measured from their average",
        fontsize=11)
    fig.text(0.995, 0.005,
             f"{FITTING_MODEL} vs {REFERENCE_MODEL} calibration re-fitted per scan; "
             f"NA 0.42 post-hoc, D = {SESSION_D_MM} mm, n = {N_SAMPLE}",
             ha="right", va="bottom", fontsize=7, color="0.45")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    out = OUT / "cornea_sweep_20260821.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n-> {out}")

    gate_sensitivity(all_rows)
