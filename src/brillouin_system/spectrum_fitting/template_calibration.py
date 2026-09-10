"""Non-parametric calibration chain: line centres AND line shapes from the
measured profile, with no instrument model (no sigma, no tau, no Lorentzian).

The chain itself lives in spectrum_fitting/epsf.py (class Epsf, which owns
the node table, the template fit, the per-position kernel lookup and the
sine diagnostic). This module is the calibration-side entry point: it turns
a CalibrationData into CalibrationPolyfitParameters whose tracks are
TEMPLATE centres, and hands the Epsf along for the sample kernels.

Judged on the once-per-pixel sine of the calibration-line centres: 0.05-0.13
MHz on three 401-point sweeps (parametric reference 0.02-0.32), 0.08-0.15 on
a single 41-point calibration (2026-09-07). Template and parametric centres
differ by a CONSTANT convention offset (+0.34 px left, +0.15 px right) that
cancels between calibration and sample as long as one convention is used
throughout — which is why the Epsf owns both the axis and the kernel.
"""
from __future__ import annotations

import numpy as np

from brillouin_system.spectrum_fitting.epsf import (
    Epsf, LINE_NAMES, SMOOTH_DEGREE, _hwhm)

# the name the fitter, the axial-scan chain and older analysis scripts use
TemplateProfiles = Epsf


def build_template_calibration(calibration_data, fitter, n_lines: int):
    """Run the chain. Returns (frames, Epsf) with the final template centres
    on each frame (frames is the Epsf's own frame list)."""
    epsf = Epsf.from_calibration(calibration_data, fitter, n_lines)
    return epsf.frames, epsf


def calibration_parameters_from_template(calibration_data, fitter, n_lines,
                                         degree, outer_degree=None):
    """CalibrationPolyfitParameters with every track from TEMPLATE centres,
    plus the Epsf for the sample kernels. outer_degree = degree of the
    outer-order frequency tracks (None = the live calibration config)."""
    from brillouin_system.calibration.calibration import (
        CalibrationPolyfitParameters, sort_xy, resolve_outer_degree)

    outer_deg = resolve_outer_degree(outer_degree)

    frames, tp = build_template_calibration(calibration_data, fitter, n_lines)
    fq = tp.freqs
    cen = tp.centres                                     # frames x lines
    names = LINE_NAMES[n_lines]
    iL, iR = names.index("left"), names.index("right")

    def fit(x, y, deg=degree):
        if len(x) <= deg:
            return np.full(deg + 1, np.nan)
        return np.polyfit(x, y, deg)

    def width_poly(line):
        # the template HWHM along the track (the Thompson chain and the
        # Lorentzian models read it; the DHO through the kernel does not)
        nodes = tp.nodes[line]
        w = np.array([_hwhm(tp.grid, p) for p in tp.profiles[line]])
        return fit(nodes, w)

    left, right = cen[:, iL], cen[:, iR]
    dist = right - left
    lp, lf = sort_xy(left, fq)
    rp, rf = sort_xy(right, fq)
    dp, df = sort_xy(dist, fq)
    params = CalibrationPolyfitParameters(
        degree=degree, outer_degree=outer_deg,
        freq_left_peak=fit(left, fq), freq_right_peak=fit(right, fq),
        freq_peak_distance=fit(dist, fq),
        calibration_width_left_peak=width_poly(iL),
        calibration_width_right_peak=width_poly(iR),
        left_px_points=lp, left_freq_points=lf,
        right_px_points=rp, right_freq_points=rf,
        dist_px_points=dp, dist_freq_points=df,
    )
    if n_lines == 4:
        iOL, iOR = names.index("outer_left"), names.index("outer_right")
        ol, orr = cen[:, iOL], cen[:, iOR]
        params.freq_outer_left_peak = fit(ol, fq, outer_deg)
        params.freq_outer_right_peak = fit(orr, fq, outer_deg)
        params.calibration_width_outer_left_peak = width_poly(iOL)
        params.calibration_width_outer_right_peak = width_poly(iOR)
        params.outer_left_px_points, params.outer_left_freq_points = sort_xy(ol, fq)
        params.outer_right_px_points, params.outer_right_freq_points = sort_xy(orr, fq)
        od = orr - ol                       # the outer pair's own distance track
        params.freq_outer_peak_distance = fit(od, fq, outer_deg)
        params.outer_dist_px_points, params.outer_dist_freq_points = sort_xy(od, fq)
    return params, tp


def sine_mhz(frames, line, degree=SMOOTH_DEGREE):
    """Once-per-pixel wobble of a line's centres on this calibration [MHz]:
    (sine amplitude, residual sd). `frames` is an Epsf or its frame list."""
    if isinstance(frames, Epsf):
        return frames.sine_mhz(line, degree)
    fq = np.array([f.freq for f in frames])
    cs = np.array([f.centre[line] for f in frames])
    return Epsf.sine_of(fq, cs, degree)


def save_epsf_file(calibration_h5, out_csv, fitter=None, n_lines=None):
    """Build the Epsf of a stored calibration (e.g. a 401-point fine sweep,
    Data/2026-9-7/NA014/calibration_401.h5) under the fitter's live config
    and write its node table (CSV) for kernel_source = "file". Returns the Epsf.

    The table carries the sweep's own envelope slopes (divided out before
    stacking) as provenance only; a scan fitted with it keeps its own
    envelope. Refresh the file after every realignment (2026-09-09 rule)."""
    from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
    from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
        dict_to_dataclass_tree, load_dict_from_hdf5)
    from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

    cal = dict_to_dataclass_tree(load_dict_from_hdf5(str(calibration_h5)), known_classes)
    if not hasattr(cal, "measured_freqs"):
        raise ValueError(f"{calibration_h5} does not hold a CalibrationData.")
    sf = fitter if fitter is not None else SpectrumFitter()
    epsf = Epsf.from_calibration(cal, sf, n_lines)
    epsf.save(out_csv, source=str(calibration_h5))
    return epsf


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("usage: python -m brillouin_system.spectrum_fitting."
                 "template_calibration <calibration.h5> <out.csv>")
    e = save_epsf_file(sys.argv[1], sys.argv[2])
    for line, nm in enumerate(e.names):
        print(f"{nm:12s} nodes {e.nodes[line].min():.1f}..{e.nodes[line].max():.1f} "
              f"({len(e.nodes[line])}), frames/node {int(np.median(e.node_frames[line]))}, "
              f"sine {e.sine_mhz(line)[0]:.2f} MHz")
