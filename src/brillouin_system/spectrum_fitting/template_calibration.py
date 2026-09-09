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
                                         degree):
    """CalibrationPolyfitParameters with every track from TEMPLATE centres,
    plus the Epsf for the sample kernels."""
    from brillouin_system.calibration.calibration import (
        CalibrationPolyfitParameters, sort_xy)

    frames, tp = build_template_calibration(calibration_data, fitter, n_lines)
    fq = tp.freqs
    cen = tp.centres                                     # frames x lines
    names = LINE_NAMES[n_lines]
    iL, iR = names.index("left"), names.index("right")

    def fit(x, y):
        if len(x) <= degree:
            return np.full(degree + 1, np.nan)
        return np.polyfit(x, y, degree)

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
        degree=degree,
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
        params.freq_outer_left_peak = fit(ol, fq)
        params.freq_outer_right_peak = fit(orr, fq)
        params.calibration_width_outer_left_peak = width_poly(iOL)
        params.calibration_width_outer_right_peak = width_poly(iOR)
        params.outer_left_px_points, params.outer_left_freq_points = sort_xy(ol, fq)
        params.outer_right_px_points, params.outer_right_freq_points = sort_xy(orr, fq)
    return params, tp


def sine_mhz(frames, line, degree=SMOOTH_DEGREE):
    """Once-per-pixel wobble of a line's centres on this calibration [MHz]:
    (sine amplitude, residual sd). `frames` is an Epsf or its frame list."""
    if isinstance(frames, Epsf):
        return frames.sine_mhz(line, degree)
    fq = np.array([f.freq for f in frames])
    cs = np.array([f.centre[line] for f in frames])
    return Epsf.sine_of(fq, cs, degree)
