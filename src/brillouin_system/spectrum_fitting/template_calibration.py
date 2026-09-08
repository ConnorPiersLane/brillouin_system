"""Non-parametric calibration chain: line centres AND line shapes from the
measured profile, with no instrument model (no sigma, no tau, no Lorentzian).

Two jobs, both done here from the scan's own calibration frames:

  centres  -> the px->GHz polynomials (the frequency axis, "where the line is")
  shapes   -> the instrument profile at any position along each line's track
              (the DHO sample kernel, "what shape the line has")

Recipe (validated 2026-09-07, Data/2026-9-7/nonparam_chain_97.py and
nonparam_e2e_97.py; memory nonparametric-chain-0907):

 1. every calibration frame: the n_peaks brightest lines are located and each
    is fitted with a PLAIN pixel-sampled Lorentzian (first guess only; its
    centres wobble once per pixel by 16-23 MHz, which is irrelevant because
    of step 2);
 2. the stacking reference is NOT those centres but a CUBIC IN DRIVE
    FREQUENCY through them: a once-per-pixel wobble is a sine in pixel
    phase, orthogonal to a low-order polynomial in f, so it drops out.
    (Stacking on wobbling centres warps the profile coherently and a naive
    iteration never converges — the trap the August work fell into.)
 3. stack: other lines subtracted with their Lorentzian first guesses (they
    are > 30 px away), floor from the peak-free frame ends, each frame
    normalised by its AREA above the floor (model-free; a fitted amplitude
    of a wrong model is phase-dependent), local-QUADRATIC smoother with
    h = 0.12 px (a wider bandwidth broadens the template and puts a
    phase-dependent error back into the centres), nodes every 1 px along
    the track, +-2.5 px window each;
 4. every frame is refitted with the nearest-node profile as the TEMPLATE
    (cubic spline in u; amplitude, centre, offset free) -> template centres;
 5. one more pass of 2-4 (it converges after one);
 6. calibration polynomials (production degree) from the template centres;
    the "width" polynomials carry the template HWHM along the track (the
    Thompson chain and the Lorentzian models read them; the DHO through the
    measured kernel does not).

Judged on the once-per-pixel sine of the calibration-line centres: 0.05-0.13
MHz on three 401-point sweeps (parametric reference 0.02-0.32), 0.08-0.15 on
a single 41-point calibration; the water fit on this axis reproduces the
production (parametric-centre) chain within 0.3 MHz in shift and 0.005 in
width/Holmes, with a smaller left-right gap. Template and parametric centres
differ by a CONSTANT convention offset (+0.34 px left, +0.15 px right) that
cancels between calibration and sample as long as one convention is used
throughout — which is why this chain must own both the axis and the kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from brillouin_system.spectrum_fitting.measured_kernel import (
    KERNEL_HALF_PX, MeasuredKernel, SMOOTH_H_PX, WINDOW_PX, _local_quadratic)
from brillouin_system.spectrum_fitting.psf import DX

SEED_WINDOW_PX = 6.0        # plain-Lorentzian first guess and template fits
FLOOR_FRACTION = 0.1        # frame floor from the peak-free axis ends
SMOOTH_DEGREE = 3           # centre reference: cubic in drive frequency
NODE_STEP_PX = 1.0
N_PASSES = 2
LINE_NAMES = {2: ("left", "right"),
              4: ("outer_left", "left", "right", "outer_right")}


def _lorentzian(x, a, c, g, o):
    return o + a * g ** 2 / ((x - c) ** 2 + g ** 2)


def _hwhm(u, p):
    i = int(np.argmax(p))
    half = p[i] / 2
    left = u[:i][p[:i] <= half]
    right = u[i:][p[i:] <= half]
    lo = left.max() if left.size else u[0]
    hi = right.min() if right.size else u[-1]
    return float((hi - lo) / 2)


@dataclass
class _Frame:
    px: np.ndarray
    sline: np.ndarray
    freq: float
    seed: list            # per line: (amp, centre, hwhm) plain-Lorentzian guess
    centre: np.ndarray    # per line: current centre estimate (template fit)


@dataclass
class TemplateProfiles:
    """The stacked instrument profiles of one calibration, per line, plus the
    frame data needed to build a kernel at any position along the track."""
    n_lines: int
    grid: np.ndarray                      # u [px], step DX, +-KERNEL_HALF_PX
    nodes: list = field(default_factory=list)      # per line: node positions [px]
    profiles: list = field(default_factory=list)   # per line: {node: unit-area profile}
    # VIPA envelope slope per line [1/px] (sline config) that was DIVIDED OUT
    # of every frame before stacking, so the profile is the instrument alone
    # and the sample fit applies the envelope exactly once, as production
    # does with the parametric kernel. (Leaving it in and applying it again
    # put the outer orders 6 MHz off; dropping it on the sample side put
    # them 3-8 MHz the other way — measured 2026-09-07 on scan 7.)
    envs: list = field(default_factory=list)
    # per-scan EnvelopeModel (envelope_source = "measured"); None = constants
    envelope: object | None = None
    _frames: list = field(default_factory=list, repr=False)
    _smooth_centres: np.ndarray | None = field(default=None, repr=False)

    def env_slope(self, line: int, x: float) -> float:
        """The envelope slope applied for `line` at position x."""
        if self.envelope is not None:
            return float(self.envelope.slope(x))
        return float(self.envs[line]) if self.envs else 0.0

    @property
    def names(self):
        return LINE_NAMES[self.n_lines]

    def kernel_at(self, line: int, position_px: float) -> MeasuredKernel:
        """The measured instrument kernel of `line` at `position_px`, stacked
        from the frames whose (frequency-smoothed) centre lies within
        +-WINDOW_PX of it — the same stack the centres came from."""
        k, n = _stack(self._frames, self._smooth_centres, line, position_px,
                      self.grid, self.env_slope(line, position_px))
        return MeasuredKernel(u=self.grid, k=k, position_px=float(position_px),
                              n_frames=n, g_median_px=_hwhm(self.grid, k))

    def hwhm_px(self, line: int, position_px: float) -> float:
        return float(_hwhm(self.grid, self.kernel_at(line, position_px).k))


def _locate_lines(px, sline, n_lines):
    floor = np.median(sline)
    peaks, props = find_peaks(sline, prominence=0.05 * (sline.max() - floor),
                              distance=8)
    if len(peaks) < n_lines:
        raise ValueError(f"Only {len(peaks)} lines found in a calibration "
                         f"frame, {n_lines} needed.")
    order = np.argsort(props["prominences"])[::-1][:n_lines]
    return np.sort(px[peaks[order]])


def _seed_fit(px, sline, c0):
    m = np.abs(px - c0) <= SEED_WINDOW_PX
    x, y = px[m], sline[m]
    p, _ = curve_fit(_lorentzian, x, y,
                     p0=[y.max() - y.min(), c0, 0.6, y.min()], maxfev=4000)
    return float(p[0]), float(p[1]), float(abs(p[2]))


def _read_frames(calibration_data, fitter, n_lines):
    frames = []
    for block in calibration_data.measured_freqs:
        for point in block.cali_meas_points:
            px, sline = fitter.get_px_sline_from_image(
                np.asarray(point.frame, dtype=float))
            px = np.asarray(px, dtype=float)
            sline = np.asarray(sline, dtype=float)
            try:
                c0s = _locate_lines(px, sline, n_lines)
                seed = [_seed_fit(px, sline, c0) for c0 in c0s]
            except (ValueError, RuntimeError):
                continue
            frames.append(_Frame(px, sline, float(point.microwave_freq), seed,
                                 np.array([s[1] for s in seed])))
    frames.sort(key=lambda f: f.freq)
    if len(frames) < SMOOTH_DEGREE + 3:
        raise ValueError(f"Only {len(frames)} usable calibration frames.")
    return frames


def _smooth_centres(frames, n_lines):
    fq = np.array([f.freq for f in frames])
    out = np.zeros((len(frames), n_lines))
    for line in range(n_lines):
        cs = np.array([f.centre[line] for f in frames])
        out[:, line] = np.polyval(np.polyfit(fq, cs, SMOOTH_DEGREE), fq)
    return out


def _cleaned(frame, line):
    """This line's counts above the floor with the other lines subtracted
    (their plain-Lorentzian first guesses), and its area."""
    y = frame.sline.copy()
    for j, (a, c, g) in enumerate(frame.seed):
        if j != line:
            y = y - _lorentzian(frame.px, a, c, g, 0.0)
    n_floor = max(int(round(FLOOR_FRACTION * len(frame.px))), 3)
    y = y - float(np.median(np.r_[y[:n_floor], y[-n_floor:]]))
    return y


def _stack(frames, smooth, line, position, grid, env=0.0):
    U, Y, n = [], [], 0
    for i, fr in enumerate(frames):
        c = smooth[i, line]
        if abs(c - position) > WINDOW_PX:
            continue
        y = _cleaned(fr, line)
        u = fr.px - c
        if env != 0.0:
            # take the multiplicative VIPA envelope out: the stacked profile
            # is then the instrument response alone (see TemplateProfiles)
            y = y / np.exp(env * u)
        w = np.abs(u) <= KERNEL_HALF_PX + 0.5
        area = float(np.sum(y[w]))
        if not area > 0:
            continue
        U.extend(u[w].tolist())
        Y.extend((y[w] / area).tolist())
        n += 1
    if n < 8:
        raise ValueError(f"Only {n} calibration frames within +-{WINDOW_PX} "
                         f"px of px {position:.1f} for line {line}.")
    k = np.clip(_local_quadratic(np.asarray(U), np.asarray(Y), grid,
                                 SMOOTH_H_PX), 0.0, None)
    return k / float(k.sum() * DX), n


def _node_profiles(frames, smooth, line, grid, env_at):
    cs = smooth[:, line]
    nodes = np.arange(np.floor(cs.min()) + WINDOW_PX,
                      cs.max() - WINDOW_PX + 0.01, NODE_STEP_PX)
    profiles = {}
    for node in nodes:
        try:
            profiles[float(node)], _ = _stack(frames, smooth, line, node, grid,
                                              env_at(node))
        except ValueError:
            continue
    if not profiles:
        raise ValueError(f"No node profile could be built for line {line}.")
    return profiles


def _template_centres(frames, profiles, line, grid, env_at):
    nodes = np.array(sorted(profiles))
    for fr in frames:
        c0 = fr.centre[line]
        p = profiles[float(nodes[np.argmin(np.abs(nodes - c0))])]
        spline = CubicSpline(grid, p / p.max(), extrapolate=False)
        env = env_at(c0)

        def template(x, a, c, o):
            t = np.nan_to_num(spline(x - c), nan=0.0)
            if env != 0.0:
                t = t * np.exp(env * (x - c))
            return o + a * t
        m = np.abs(fr.px - c0) <= SEED_WINDOW_PX
        x, y = fr.px[m], fr.sline[m]
        try:
            popt, _ = curve_fit(template, x, y,
                                p0=[y.max() - y.min(), c0, y.min()], maxfev=4000)
            fr.centre[line] = float(popt[1])
        except RuntimeError:
            pass


def build_template_calibration(calibration_data, fitter, n_lines: int):
    """Run the chain. Returns (frames, TemplateProfiles) with the final
    template centres on each frame."""
    grid = np.arange(-KERNEL_HALF_PX, KERNEL_HALF_PX + DX / 2, DX)
    frames = _read_frames(calibration_data, fitter, n_lines)
    sl = fitter.sline_config
    if n_lines == 4:
        envs = [float(sl.env_slope_outer_left_perpx), float(sl.env_slope_left_perpx),
                float(sl.env_slope_right_perpx), float(sl.env_slope_outer_right_perpx)]
    else:
        envs = [float(sl.env_slope_left_perpx), float(sl.env_slope_right_perpx)]
    envelope = None
    if getattr(sl, "envelope_source", "config") == "measured":
        from brillouin_system.spectrum_fitting.envelope import envelope_from_calibration
        try:
            envelope = envelope_from_calibration(calibration_data, fitter)
        except ValueError as e:
            import logging
            logging.getLogger(__name__).warning(
                f"[template_calibration] {e} — using the config envelope slopes.")
    tp = TemplateProfiles(n_lines=n_lines, grid=grid, envs=envs, envelope=envelope)
    for _ in range(N_PASSES):
        smooth = _smooth_centres(frames, n_lines)
        tp.nodes, tp.profiles = [], []
        for line in range(n_lines):
            env_at = (lambda x, line=line: tp.env_slope(line, x))
            prof = _node_profiles(frames, smooth, line, grid, env_at)
            tp.nodes.append(np.array(sorted(prof)))
            tp.profiles.append(prof)
            _template_centres(frames, prof, line, grid, env_at)
    tp._frames = frames
    tp._smooth_centres = _smooth_centres(frames, n_lines)
    return frames, tp


def calibration_parameters_from_template(calibration_data, fitter, n_lines,
                                         degree):
    """CalibrationPolyfitParameters with every track from TEMPLATE centres,
    plus the TemplateProfiles for the sample kernels."""
    from brillouin_system.calibration.calibration import (
        CalibrationPolyfitParameters, sort_xy)

    frames, tp = build_template_calibration(calibration_data, fitter, n_lines)
    fq = np.array([f.freq for f in frames])
    cen = np.array([f.centre for f in frames])          # frames x lines
    names = LINE_NAMES[n_lines]
    iL, iR = names.index("left"), names.index("right")

    def fit(x, y):
        if len(x) <= degree:
            return np.full(degree + 1, np.nan)
        return np.polyfit(x, y, degree)

    def width_poly(line):
        nodes = tp.nodes[line]
        w = np.array([_hwhm(tp.grid, tp.profiles[line][float(n)]) for n in nodes])
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
    the diagnostic of the centre job."""
    fq = np.array([f.freq for f in frames])
    cs = np.array([f.centre[line] for f in frames])
    poly = np.polyfit(fq, cs, degree)
    res = cs - np.polyval(poly, fq)
    disp = 1e3 / abs(float(np.polyval(np.polyder(poly), np.median(fq))))
    ph = 2 * np.pi * (cs - np.floor(cs))
    A = np.column_stack([np.sin(ph), np.cos(ph), np.ones_like(ph)])
    coef, *_ = np.linalg.lstsq(A, res, rcond=None)
    return float(np.hypot(coef[0], coef[1]) * disp), float(res.std() * disp)
