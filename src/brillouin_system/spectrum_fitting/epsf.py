"""The measured instrument response (effective PSF, "ePSF") of one calibration.

One object owns BOTH jobs of the self-calibrating chain, from the same
calibration frames, in the same centre convention:

  the AXIS    every calibration line is fitted with the measured profile as
              the template -> template centres -> the px->GHz polynomials;
  the KERNEL  the measured profile at any position along each line's track
              -> the DHO sample kernel (dho.dho_profile(kernel=...)).

Why one object. The centre of an asymmetric line is a convention. The
parametric model puts it at the Lorentzian core, the template puts it at
u = 0 of the stacked profile, and the two differ by a constant (+0.34 px
left, +0.15 px right, measured 2026-09-07). The offset cancels between the
calibration axis and the sample fit ONLY when both use the same profile.
Mixing conventions is 0.19 px on the distance, ~55 MHz on the shift, and
nothing downstream would notice. Hence the axis and the kernel come out of
one Epsf and nowhere else.

Inputs are spectra, not images and not a fitter: per calibration frame the
drive frequency, the pixel axis, and the row-summed spectrum. The image ->
spectrum step (row band) belongs to the fitter; Epsf.from_calibration wraps
it.

Recipe (validated 2026-09-07, memory nonparametric-chain-0907):

 1. per frame the n_lines brightest lines are located and each is fitted
    with a PLAIN pixel-sampled Lorentzian (first guess only; its centres
    wobble once per pixel by 16-23 MHz, irrelevant because of 2);
 2. the stacking reference is NOT those centres but a CUBIC IN DRIVE
    FREQUENCY through them: a once-per-pixel wobble is a sine in pixel
    phase, orthogonal to a low-order polynomial in f, so it drops out.
    Stacking on the wobbling centres warps the profile coherently and a
    naive iteration never converges (the August trap);
 3. stack: the other lines subtracted with their Lorentzian first guesses
    (they are > 30 px away), floor from the peak-free frame ends, the VIPA
    envelope divided out, each frame normalised by its AREA above the floor
    (model-free), local-QUADRATIC smoother with h = 0.12 px on a 0.02 px
    grid, one profile per NODE every 1 px along the track from the frames
    within +-2.5 px;
 4. every frame is refitted with the profile as the template (cubic spline
    in u, blended between the two nodes bracketing the centre, amplitude /
    centre / offset free) -> template centres;
 5. one more pass of 2-4 (it converges after one).

Two conventions the consumers rely on:
  * the profile is measured ON the pixel grid, so it already contains the
    pixel box. A convolution with it must not add another one
    (dho_profile does not).
  * the VIPA envelope is divided OUT before stacking, so the profile is the
    instrument alone and the caller applies the envelope exactly once
    (__call__ does; the fitter does for dho_profile).

Storage: per line a node ladder `nodes[line]` (px), a profile table
`profiles[line]` of shape (n_nodes, n_grid) on the shared `grid` (u in px,
step DX, +-KERNEL_HALF_PX, unit area each), and the frame count per node.
A position between two nodes gets the linear blend of the two bracketing
profiles by distance (the end profile alone beyond the ladder, an error
beyond +-WINDOW_PX outside it). Adjacent nodes differ by ~1 % of the peak,
so the blend mostly averages stack noise and keeps the kernel continuous
as a peak walks along a scan.

Stored tables (save/load) and FILE kernels: a table built on a fine sweep
(401 points, 0.01 GHz) can serve as the SAMPLE KERNEL of a scan whose own
41-point sweep under-samples a line (the outer-left line steps 0.54 px per
frame there, a half-pixel alias, 2026-09-09). FileKernels below pairs such
a table with the scan's own Epsf: the kernels of the named lines come from
the file, the axis, the other kernels and the envelope from the scan. That
is a deliberate mix of two profiles' centre conventions (see above), so it
is only valid once the inner distance has been checked against the
per-scan kernels (< 0.3 MHz on the reference scans, Data/2026-9-9).

Diagnostic: the once-per-pixel sine of the centres about a cubic in drive
frequency (sine_mhz). In-sample the cubic was the stacking reference, so
the centres following it is partly circular. The honest number is HELD
OUT: build on one set of frames, fit another (held_out_sine).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

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
MIN_NODE_FRAMES = 8
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


@dataclass
class Frame:
    """One calibration frame as the chain sees it."""
    px: np.ndarray
    sline: np.ndarray
    freq: float
    seed: list            # per line: (amp, centre, hwhm) plain-Lorentzian guess
    centre: np.ndarray    # per line: current centre estimate (template fit)

    @classmethod
    def read(cls, px, sline, freq, n_lines):
        px = np.asarray(px, dtype=float)
        sline = np.asarray(sline, dtype=float)
        c0s = _locate_lines(px, sline, n_lines)
        seed = [_seed_fit(px, sline, c0) for c0 in c0s]
        return cls(px, sline, float(freq), seed, np.array([s[1] for s in seed]))

    def cleaned(self, line):
        """This line's counts above the floor with the other lines subtracted
        (their plain-Lorentzian first guesses)."""
        y = self.sline.copy()
        for j, (a, c, g) in enumerate(self.seed):
            if j != line:
                y = y - _lorentzian(self.px, a, c, g, 0.0)
        n_floor = max(int(round(FLOOR_FRACTION * len(self.px))), 3)
        y = y - float(np.median(np.r_[y[:n_floor], y[-n_floor:]]))
        return y


def read_frames(freqs, pxs, slines, n_lines):
    """Frames from raw spectra, sorted by drive frequency; unreadable frames
    (lines not found, seed fit failed) are dropped."""
    frames = []
    for f, px, s in zip(freqs, pxs, slines):
        try:
            frames.append(Frame.read(px, s, f, n_lines))
        except (ValueError, RuntimeError):
            continue
    frames.sort(key=lambda fr: fr.freq)
    return frames


@dataclass
class HeldOutSine:
    """Once-per-pixel sine [MHz] per line: `train` in-sample (partly
    circular), `test` on frames the ePSF was not built from, `seed` the
    plain-Lorentzian first-guess centres on the test frames (the wobble the
    template is supposed to remove). Each entry (amplitude, residual sd)."""
    names: tuple
    train: list
    test: list
    seed: list
    n_train: int
    n_test: int


class Epsf:
    """The measured instrument response of one calibration, per line, at
    any position along the line's track. See the module docstring."""

    def __init__(self, freqs, pxs, slines, n_lines: int, env_slope=None, *,
                 envelope=None, n_passes: int = N_PASSES):
        """freqs      drive frequency per frame [GHz]
        pxs        pixel axis per frame
        slines     row-summed spectrum per frame
        n_lines    2 (inner pair) or 4 (with the outer orders)
        env_slope  VIPA envelope slope [1/px] divided out before stacking:
                   None (0), a per-line sequence of constants, or a callable
                   (line, x) -> slope
        envelope   optional per-scan EnvelopeModel; when given its slope(x)
                   is used for every line and it is exposed as .envelope for
                   the fitter (per-frame lookup)
        """
        self.n_lines = int(n_lines)
        self.grid = np.arange(-KERNEL_HALF_PX, KERNEL_HALF_PX + DX / 2, DX)
        self.envelope = envelope
        self._env = env_slope
        self.source = ""            # provenance of a stored table (save/load)
        self.path = None
        self.frames = read_frames(freqs, pxs, slines, self.n_lines)
        if len(self.frames) < SMOOTH_DEGREE + 3:
            raise ValueError(f"Only {len(self.frames)} usable calibration frames.")
        self.nodes: list[np.ndarray] = []
        self.profiles: list[np.ndarray] = []
        self.node_frames: list[np.ndarray] = []
        self._splines: list[list[CubicSpline]] = []
        self.smooth_centres = None
        for _ in range(int(n_passes)):
            self._pass()
        self.smooth_centres = self._smooth(self.frames)

    # ----------------------------------------------------------- building
    @classmethod
    def from_calibration(cls, calibration_data, fitter, n_lines=None, **kw):
        """Epsf from a CalibrationData through the fitter's image -> spectrum
        step and its envelope settings (sline config constants, or the
        per-scan EnvelopeModel when envelope_source = "measured")."""
        sl = fitter.sline_config
        n_lines = int(sl.n_peaks) if n_lines is None else int(n_lines)
        freqs, pxs, slines = [], [], []
        for block in calibration_data.measured_freqs:
            for point in block.cali_meas_points:
                px, s = fitter.get_px_sline_from_image(
                    np.asarray(point.frame, dtype=float))
                freqs.append(float(point.microwave_freq))
                pxs.append(np.asarray(px, dtype=float))
                slines.append(np.asarray(s, dtype=float))
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
                    f"[epsf] {e} — using the config envelope slopes.")
        return cls(freqs, pxs, slines, n_lines, env_slope=envs,
                   envelope=envelope, **kw)

    @classmethod
    def from_table(cls, grid, nodes, profiles, node_frames, env_slope=None,
                   envelope=None):
        """An Epsf from a stored node table (no frames, no refit): `nodes`
        per line, `profiles` per line as (n_nodes, n_grid) arrays or
        {node: profile} dicts, `node_frames` per line as arrays or dicts."""
        self = cls.__new__(cls)
        self.n_lines = len(nodes)
        self.source = ""
        self.path = None
        self.grid = np.asarray(grid, dtype=float)
        self.envelope = envelope
        self._env = env_slope
        self.frames = []
        self.smooth_centres = None
        self.nodes, self.profiles, self.node_frames = [], [], []
        for nd, pr, nf in zip(nodes, profiles, node_frames):
            nd = np.asarray(nd, dtype=float)
            if isinstance(pr, dict):
                pr = np.array([pr[float(n)] for n in nd])
            if isinstance(nf, dict):
                nf = np.array([nf.get(float(n), 0) for n in nd])
            order = np.argsort(nd)
            self.nodes.append(nd[order])
            self.profiles.append(np.asarray(pr, dtype=float)[order])
            self.node_frames.append(np.asarray(nf)[order])
        self._splines = [[CubicSpline(self.grid, p, extrapolate=False) for p in pr]
                         for pr in self.profiles]
        return self

    def _smooth(self, frames):
        """Per-frame, per-line reference centres: a cubic in drive frequency
        through the current centres (the stacking reference)."""
        fq = np.array([f.freq for f in frames])
        out = np.zeros((len(frames), self.n_lines))
        for line in range(self.n_lines):
            cs = np.array([f.centre[line] for f in frames])
            out[:, line] = np.polyval(np.polyfit(fq, cs, SMOOTH_DEGREE), fq)
        return out

    def _stack(self, smooth, line, position):
        U, Y, n = [], [], 0
        env = self.env_slope(line, position)
        for i, fr in enumerate(self.frames):
            c = smooth[i, line]
            if abs(c - position) > WINDOW_PX:
                continue
            y = fr.cleaned(line)
            u = fr.px - c
            if env != 0.0:
                y = y / np.exp(env * u)
            w = np.abs(u) <= KERNEL_HALF_PX + 0.5
            area = float(np.sum(y[w]))
            if not area > 0:
                continue
            U.extend(u[w].tolist())
            Y.extend((y[w] / area).tolist())
            n += 1
        if n < MIN_NODE_FRAMES:
            raise ValueError(f"Only {n} calibration frames within +-{WINDOW_PX} "
                             f"px of px {position:.1f} for line {line}.")
        k = np.clip(_local_quadratic(np.asarray(U), np.asarray(Y), self.grid,
                                     SMOOTH_H_PX), 0.0, None)
        return k / float(k.sum() * DX), n

    def _pass(self):
        smooth = self._smooth(self.frames)
        self.nodes, self.profiles, self.node_frames, self._splines = [], [], [], []
        for line in range(self.n_lines):
            cs = smooth[:, line]
            ladder = np.arange(np.floor(cs.min()) + WINDOW_PX,
                               cs.max() - WINDOW_PX + 0.01, NODE_STEP_PX)
            nodes, profiles, counts = [], [], []
            for node in ladder:
                try:
                    k, n = self._stack(smooth, line, float(node))
                except ValueError:
                    continue
                nodes.append(float(node))
                profiles.append(k)
                counts.append(n)
            if not nodes:
                raise ValueError(f"No node profile could be built for line {line}.")
            self.nodes.append(np.array(nodes))
            self.profiles.append(np.array(profiles))
            self.node_frames.append(np.array(counts))
            self._splines.append([CubicSpline(self.grid, p, extrapolate=False)
                                  for p in profiles])
        for fr in self.frames:
            for line in range(self.n_lines):
                try:
                    _, c, _ = self.fit_line(line, fr.px, fr.sline, fr.centre[line])
                    fr.centre[line] = c
                except (ValueError, RuntimeError):
                    pass

    # ------------------------------------------------------------ lookup
    @property
    def names(self):
        return LINE_NAMES[self.n_lines]

    def env_slope(self, line: int, x: float) -> float:
        """The envelope slope [1/px] applied for `line` at position x."""
        if self.envelope is not None:
            return float(self.envelope.slope(x))
        if self._env is None:
            return 0.0
        if callable(self._env):
            return float(self._env(line, x))
        return float(self._env[line])

    def node_at(self, line: int, position_px: float) -> float:
        """The nearest node (px) of `line`; raises when the position lies
        outside the calibration sweep's coverage (no node within
        +-WINDOW_PX), where no frames sampled the profile."""
        nodes = self.nodes[line]
        j = int(np.argmin(np.abs(nodes - float(position_px))))
        if abs(nodes[j] - float(position_px)) > WINDOW_PX:
            raise ValueError(
                f"No template node within +-{WINDOW_PX} px of px "
                f"{position_px:.1f} on line {self.names[line]} (nodes "
                f"{nodes.min():.1f}..{nodes.max():.1f}).")
        return float(nodes[j])

    def _bracket(self, line: int, x: float):
        """(index a, index b, weight t of b) of the two nodes bracketing x."""
        nodes = self.nodes[line]
        if x <= nodes[0]:
            return 0, 0, 0.0
        if x >= nodes[-1]:
            return len(nodes) - 1, len(nodes) - 1, 0.0
        j = int(np.searchsorted(nodes, x, side="right")) - 1
        return j, j + 1, (x - nodes[j]) / (nodes[j + 1] - nodes[j])

    def kernel(self, line: int, position_px: float) -> MeasuredKernel:
        """The unit-area profile of `line` at `position_px` on the fine grid,
        for convolution (dho_profile(kernel=...)). Linear blend of the two
        bracketing node profiles."""
        self.node_at(line, position_px)              # coverage check
        a, b, t = self._bracket(line, float(position_px))
        k = (1.0 - t) * self.profiles[line][a] + t * self.profiles[line][b]
        n = int(round((1.0 - t) * self.node_frames[line][a]
                      + t * self.node_frames[line][b]))
        return MeasuredKernel(u=self.grid, k=k, position_px=float(position_px),
                              n_frames=n, g_median_px=_hwhm(self.grid, k))

    kernel_at = kernel          # name the fitter and the axial-scan chain use

    def hwhm_px(self, line: int, position_px: float) -> float:
        return float(_hwhm(self.grid, self.kernel(line, position_px).k))

    def __call__(self, line: int, cen: float, px, amp: float = 1.0,
                 offset: float = 0.0):
        """The line profile sampled at pixels `px` for a line centred at
        `cen`: unit PEAK, times `amp`, plus `offset`, with the envelope
        applied once. The profile is the blend at `cen`, so it follows the
        centre as a fit moves it."""
        px = np.asarray(px, dtype=float)
        a, b, t = self._bracket(line, float(cen))
        sa, sb = self._splines[line][a], self._splines[line][b]
        u = px - float(cen)
        prof = (1.0 - t) * np.nan_to_num(sa(u), nan=0.0)
        if t != 0.0:
            prof = prof + t * np.nan_to_num(sb(u), nan=0.0)
        peak = ((1.0 - t) * self.profiles[line][a].max()
                + t * self.profiles[line][b].max())
        prof = prof / peak
        env = self.env_slope(line, float(cen))
        if env != 0.0:
            prof = prof * np.exp(env * u)
        return offset + amp * prof

    # ------------------------------------------------------------ fitting
    def fit_line(self, line: int, px, sline, c0: float):
        """Fit one line of a spectrum with this ePSF as the template within
        +-SEED_WINDOW_PX of the first guess `c0`. Returns (amp, cen, offset).
        Beyond the node ladder the end profile is used (the last frames of a
        sweep sit up to ~3 px past the last node, which needs 8 frames within
        +-WINDOW_PX; the validated chain fits them with the end profile). The
        sample-side kernel() keeps its coverage check."""
        px = np.asarray(px, dtype=float)
        sline = np.asarray(sline, dtype=float)
        m = np.abs(px - float(c0)) <= SEED_WINDOW_PX
        x, y = px[m], sline[m]

        def model(xx, a, c, o):
            return self(line, c, xx, a, o)
        popt, _ = curve_fit(model, x, y, p0=[y.max() - y.min(), float(c0), y.min()],
                            maxfev=4000)
        return float(popt[0]), float(popt[1]), float(popt[2])

    def fit_lines(self, pxs, slines, freqs=None):
        """Template centres of spectra this ePSF was NOT necessarily built
        from: (n_frames, n_lines) array, NaN where a line could not be
        located or fitted. With `freqs` the rows come sorted by frequency
        and the sorted frequencies are returned too."""
        freqs = (np.zeros(len(pxs)) if freqs is None
                 else np.asarray(freqs, dtype=float))
        frames = read_frames(freqs, pxs, slines, self.n_lines)
        out = np.full((len(frames), self.n_lines), np.nan)
        for i, fr in enumerate(frames):
            for line in range(self.n_lines):
                try:
                    _, out[i, line], _ = self.fit_line(line, fr.px, fr.sline,
                                                        fr.centre[line])
                except (ValueError, RuntimeError):
                    pass
        return out, np.array([fr.freq for fr in frames]), frames

    @property
    def centres(self) -> np.ndarray:
        """Template centres of the frames the ePSF was built from,
        (n_frames, n_lines)."""
        return np.array([f.centre for f in self.frames])

    @property
    def freqs(self) -> np.ndarray:
        return np.array([f.freq for f in self.frames])

    # --------------------------------------------------------- diagnostics
    @staticmethod
    def sine_of(freqs, centres, degree: int = SMOOTH_DEGREE):
        """Once-per-pixel wobble of line centres [MHz]: (sine amplitude,
        residual sd) of the centres about a degree-`degree` polynomial in
        drive frequency, converted with the local dispersion."""
        fq = np.asarray(freqs, dtype=float)
        cs = np.asarray(centres, dtype=float)
        ok = np.isfinite(cs)
        fq, cs = fq[ok], cs[ok]
        poly = np.polyfit(fq, cs, degree)
        res = cs - np.polyval(poly, fq)
        disp = 1e3 / abs(float(np.polyval(np.polyder(poly), np.median(fq))))
        ph = 2 * np.pi * (cs - np.floor(cs))
        A = np.column_stack([np.sin(ph), np.cos(ph), np.ones_like(ph)])
        coef, *_ = np.linalg.lstsq(A, res, rcond=None)
        return float(np.hypot(coef[0], coef[1]) * disp), float(res.std() * disp)

    def sine_mhz(self, line: int, degree: int = SMOOTH_DEGREE):
        """In-sample sine of `line` (the frames the ePSF was built from)."""
        return self.sine_of(self.freqs, self.centres[:, line], degree)

    @classmethod
    def held_out_sine(cls, freqs, pxs, slines, n_lines: int, env_slope=None,
                      split: str = "even_odd", **kw) -> HeldOutSine:
        """Build the ePSF on one part of a sweep and fit the rest with it.

        split = "even_odd": every other frame (by frequency order) trains,
        the others test — the same instrument state, the honest version of
        the in-sample sine. For a cross-day test build an Epsf on one sweep
        and call fit_lines on the other.
        """
        freqs = np.asarray(freqs, dtype=float)
        order = np.argsort(freqs)
        if split != "even_odd":
            raise ValueError("split must be 'even_odd'")
        train = order[0::2]
        test = order[1::2]
        pick = lambda idx, seq: [seq[i] for i in idx]
        epsf = cls(freqs[train], pick(train, pxs), pick(train, slines), n_lines,
                   env_slope=env_slope, **kw)
        cen, fq, frames = epsf.fit_lines(pick(test, pxs), pick(test, slines),
                                         freqs[test])
        seed = np.array([fr.seed[l][1] for fr in frames for l in range(n_lines)]
                        ).reshape(len(frames), n_lines)
        return HeldOutSine(
            names=LINE_NAMES[n_lines],
            train=[epsf.sine_mhz(l) for l in range(n_lines)],
            test=[cls.sine_of(fq, cen[:, l]) for l in range(n_lines)],
            seed=[cls.sine_of(fq, seed[:, l]) for l in range(n_lines)],
            n_train=len(epsf.frames), n_test=len(frames))

    # ------------------------------------------------------------ storage
    def save(self, path, source: str = ""):
        """The node table to a CSV: one row per node profile, columns
        `line, node_px, n_frames, env_slope` then the profile at each grid
        point (`u=-6.00` ... `u=6.00`, unit area, step 0.02 px). A first
        comment line carries the provenance (`source`, e.g. the calibration
        file the table was built from, the frame count and the date).
        Frames are not stored. Readable with pandas (comment="#")."""
        import csv
        import datetime as _dt
        path = Path(str(path))
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write(f"# ePSF node table (brillouin_system Epsf.save) "
                     f"written {_dt.datetime.now().isoformat(timespec='seconds')}; "
                     f"source={source or '-'}; n_frames={len(self.frames)}; "
                     f"lines={','.join(self.names)}; profiles unit area, u in px\n")
            w = csv.writer(fh)
            w.writerow(["line", "node_px", "n_frames", "env_slope"]
                       + [f"u={u:.2f}" for u in self.grid])
            for line in range(self.n_lines):
                for node, prof, n in zip(self.nodes[line], self.profiles[line],
                                         self.node_frames[line]):
                    w.writerow([self.names[line], f"{node:.4f}", int(n),
                                f"{self.env_slope(line, float(node)):.6g}"]
                               + [f"{v:.8g}" for v in prof])

    @classmethod
    def load(cls, path):
        """An Epsf from a stored node table (save). Its env_slope
        interpolates the STORED slopes, i.e. the envelope of the sweep the
        table was built from — a consumer fitting another scan must take
        the envelope from that scan (FileKernels does)."""
        import csv
        path = Path(str(path))
        source = ""
        with path.open("r", newline="", encoding="utf-8") as fh:
            first = fh.readline()
            if first.startswith("#"):
                source = first[1:].strip()
                header = next(csv.reader([fh.readline()]))
            else:
                header = next(csv.reader([first]))
            rows = list(csv.reader(fh))
        if header[:4] != ["line", "node_px", "n_frames", "env_slope"] \
                or not all(h.startswith("u=") for h in header[4:]):
            raise ValueError(f"{path} is not an ePSF node table.")
        grid = np.array([float(h[2:]) for h in header[4:]])
        by_line: dict = {}
        for r in rows:
            if not r:
                continue
            by_line.setdefault(r[0], []).append(r)
        names = [nm for nm in LINE_NAMES.get(len(by_line), tuple(by_line)) if nm in by_line]
        if len(names) != len(by_line):
            raise ValueError(f"{path}: lines {list(by_line)} are not a known set.")
        nodes, profiles, counts, envs = [], [], [], []
        for nm in names:
            rs = by_line[nm]
            nodes.append(np.array([float(r[1]) for r in rs]))
            counts.append(np.array([int(float(r[2])) for r in rs]))
            envs.append(np.array([float(r[3]) for r in rs]))
            profiles.append(np.array([[float(v) for v in r[4:]] for r in rs]))
        order = [np.argsort(nd) for nd in nodes]
        nodes = [nd[o] for nd, o in zip(nodes, order)]
        envs = [ev[o] for ev, o in zip(envs, order)]

        def env_slope(line, x, nodes=nodes, envs=envs):
            return float(np.interp(x, nodes[line], envs[line]))
        e = cls.from_table(grid, nodes, profiles, counts, env_slope=env_slope)
        e.header = source                       # the whole provenance line
        fields = dict(f.strip().split("=", 1) for f in source.split(";") if "=" in f)
        e.source = fields.get("source", "")
        e.path = str(path)
        return e


# ------------------------------------------------------------ file kernels
OUTER_LINES = ("outer_left", "outer_right")
_FILE_CACHE: dict = {}
# Match thresholds of a file kernel against the scan's own profile at the
# same position (FileKernels.check): the centre offset between the two
# profiles [px] (the convention-mixing lever: 0.01 px ~ 3 MHz on an outer
# order), the HWHM ratio and the rms difference within +-2 px [% of peak].
# Set 2026-09-09 from the same-day (9-7) and next-day (9-8) comparisons on
# the water series; a value beyond them is logged as a warning by the
# axial-scan analysis, not refused.
MATCH_SHIFT_PX = 0.03
MATCH_HWHM_FRACTION = 0.06
MATCH_RMS_PERCENT = 4.0
MATCH_CORE_PX = 2.0


@dataclass
class KernelMatch:
    """How a FILE kernel compares with the SCAN's own profile of the same
    line at one position (both unit area on the fine grid)."""
    name: str
    position_px: float
    hwhm_file_px: float
    hwhm_scan_px: float
    shift_px: float            # file profile centre minus scan profile centre
    rms_percent: float         # rms(file - scan shifted) within +-MATCH_CORE_PX, % of peak
    n_frames_scan: int         # frames behind the scan's node blend (coverage)
    in_use: bool = True        # this line's sample kernel comes from the file

    @property
    def hwhm_ratio(self) -> float:
        return self.hwhm_file_px / self.hwhm_scan_px if self.hwhm_scan_px > 0 else float("nan")

    @property
    def ok(self) -> bool:
        return (abs(self.shift_px) <= MATCH_SHIFT_PX
                and abs(self.hwhm_ratio - 1.0) <= MATCH_HWHM_FRACTION
                and self.rms_percent <= MATCH_RMS_PERCENT)

    def __str__(self):
        return (f"{self.name:12s} px {self.position_px:6.1f}: HWHM file/scan "
                f"{self.hwhm_file_px:.3f}/{self.hwhm_scan_px:.3f} px "
                f"({100 * (self.hwhm_ratio - 1):+.1f} %), centre offset "
                f"{self.shift_px:+.3f} px, rms {self.rms_percent:.1f} % of peak "
                f"({self.n_frames_scan} scan frames)"
                + ("" if self.in_use else " [scan kernel in use]")
                + ("" if self.ok else "  <-- MISMATCH"))


def _profile_shift_px(u, a, b, core_px=MATCH_CORE_PX):
    """The shift d that best maps profile b onto a within +-core_px
    (least squares over d, b(u - d) interpolated): the centre offset a - b."""
    from scipy.optimize import minimize_scalar
    m = np.abs(u) <= core_px

    def cost(d):
        return float(np.sum((a[m] - np.interp(u[m] - d, u, b, left=0.0, right=0.0)) ** 2))
    r = minimize_scalar(cost, bounds=(-0.5, 0.5), method="bounded",
                        options={"xatol": 1e-4})
    return float(r.x)


def load_epsf_file(path) -> Epsf:
    """Epsf.load, once per path for the process (like the reflection
    background): a scan series reads the table one time."""
    p = Path(str(path))
    if not p.is_file():
        raise FileNotFoundError(
            f"kernel_file {p} does not exist (a stored ePSF node table "
            "written by Epsf.save is needed for kernel_source = 'file').")
    key = str(p.resolve())
    if key not in _FILE_CACHE:
        _FILE_CACHE[key] = Epsf.load(p)
    return _FILE_CACHE[key]


class FileKernels:
    """The scan's Epsf with the SAMPLE KERNEL of the named lines taken from
    a stored node table, everything else from the scan.

    kernel / kernel_at / hwhm_px   the file's profile for `file_lines`
                                   (looked up by line NAME, so a four-line
                                   file serves a two-line scan), the scan's
                                   for every other line;
    env_slope / envelope           the SCAN's — the file's stored slopes are
                                   the fine sweep's envelope and never apply;
    everything else (names, frames, centres, sine_mhz, __call__, nodes,
    profiles, grid)                the scan's, i.e. the AXIS side.

    Why the split matters: the template centre convention is a property of
    the profile (+0.34 px left / +0.16 px right from the parametric core,
    constant per line). The axis carries the scan profile's convention, a
    file kernel the file profile's; they cancel only when the two profiles
    have the same shape. Hence this object never builds an axis, and its
    use is gated on the measured inner-distance check (module docstring).
    """

    def __init__(self, scan: Epsf, file: Epsf, lines, path=None):
        self.scan = scan
        self.file = file
        self.path = str(path) if path is not None else getattr(file, "path", None)
        names = scan.names
        lines = tuple(lines)
        unknown = [nm for nm in lines if nm not in names]
        if unknown:
            raise ValueError(f"Lines {unknown} are not fitted lines {names}.")
        missing = [nm for nm in lines if nm not in file.names]
        if missing:
            raise ValueError(
                f"The kernel file carries lines {file.names}, not {missing}.")
        self.file_lines = tuple(nm for nm in names if nm in lines)
        self._file_index = {names.index(nm): file.names.index(nm)
                            for nm in self.file_lines}

    def __getattr__(self, name):
        # axis-side attributes and diagnostics: the scan's own Epsf
        # (only reached for names not defined on this class)
        if name in ("scan", "file", "path", "file_lines", "_file_index"):
            raise AttributeError(name)
        return getattr(self.scan, name)

    @property
    def names(self):
        return self.scan.names

    @property
    def n_lines(self):
        return self.scan.n_lines

    @property
    def envelope(self):
        return self.scan.envelope

    def env_slope(self, line: int, x: float) -> float:
        return self.scan.env_slope(line, x)

    def source(self, line: int) -> str:
        """'file' or 'scan': where this line's kernel comes from."""
        return "file" if line in self._file_index else "scan"

    def kernel(self, line: int, position_px: float) -> MeasuredKernel:
        j = self._file_index.get(line)
        if j is None:
            return self.scan.kernel(line, position_px)
        return self.file.kernel(j, position_px)

    kernel_at = kernel

    def hwhm_px(self, line: int, position_px: float) -> float:
        k = self.kernel(line, position_px)
        return float(_hwhm(k.u, k.k))

    def __call__(self, line, cen, px, amp=1.0, offset=0.0):
        return self.scan(line, cen, px, amp, offset)

    def check(self, positions) -> list:
        """Compare the file profile with the scan's own profile at each
        line's position (`positions` in fit order, one per fitted line),
        for EVERY line the file carries, whether or not its kernel is in
        use: the quick match check the axial-scan analysis logs, so a
        stale table (realignment since the fine sweep) shows up. On a
        line the scan under-samples (outer_left on a 41-point sweep) the
        SCAN profile is the rough one, so a mismatch there is read
        together with the well-sampled lines."""
        out = []
        for line, nm in enumerate(self.names):
            if nm not in self.file.names:
                continue
            j = self.file.names.index(nm)
            x = float(positions[line])
            kf = self.file.kernel(j, x)
            ks = self.scan.kernel(line, x)
            if kf.u.shape != ks.u.shape or not np.allclose(kf.u, ks.u):
                raise ValueError("The file and scan node tables use different grids.")
            d = _profile_shift_px(kf.u, kf.k, ks.k)
            m = np.abs(kf.u) <= MATCH_CORE_PX
            resid = kf.k[m] - np.interp(kf.u[m] - d, ks.u, ks.k, left=0.0, right=0.0)
            out.append(KernelMatch(
                name=self.names[line], position_px=x,
                hwhm_file_px=_hwhm(kf.u, kf.k), hwhm_scan_px=_hwhm(ks.u, ks.k),
                shift_px=d, rms_percent=100.0 * float(np.sqrt(np.mean(resid ** 2))) / float(kf.k.max()),
                n_frames_scan=int(ks.n_frames), in_use=line in self._file_index))
        return out

    def report(self, positions) -> str:
        head = f"file kernels ({self.path}) for {', '.join(self.file_lines)} vs this scan's profile:"
        return "\n".join([head] + ["  " + str(m) for m in self.check(positions)])


def kernels_for_fit(scan_epsf, sline_config):
    """The kernel source for the SAMPLE fits of a scan under `sline_config`:
    the scan's own Epsf (kernel_source = "scan"), or a FileKernels pairing
    it with the stored table `kernel_file` for `kernel_file_lines`
    ("outer" or "all"). A two-line fit with "outer" has nothing to take
    from the file and keeps the scan kernels (logged)."""
    src = getattr(sline_config, "kernel_source", "scan")
    if src != "file":
        return scan_epsf
    which = getattr(sline_config, "kernel_file_lines", "outer")
    lines = OUTER_LINES if which == "outer" else tuple(scan_epsf.names)
    lines = tuple(nm for nm in lines if nm in scan_epsf.names)
    if not lines:
        logging.getLogger(__name__).warning(
            "[epsf] kernel_source = 'file' with kernel_file_lines = "
            f"'{which}' but the fit has lines {scan_epsf.names} only — "
            "using the scan's own kernels.")
        return scan_epsf
    path = getattr(sline_config, "kernel_file", "")
    return FileKernels(scan_epsf, load_epsf_file(path), lines, path=path)
