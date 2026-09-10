"""Per-scan VIPA envelope from the calibration sweep — model-free.

The VIPA transmission T(x) varies across the detector (the envelope) and
multiplies every line. The code works with its logarithm g(x) = ln T(x),
because the measurement gives area RATIOS (differences of g) and the fit
is then linear. A line of finite width is tilted by the relative slope
T'(x)/T(x) = g'(x), which is what `slope` returns; `transmission` (=
`__call__`) returns T itself, normalised at a reference pixel, for
anything that compares intensities at two positions.

For the inner pair the relative slope is ~0.01-0.02 per px
and moves the fitted DHO resonance by ~2 MHz; for the outer orders it is
~0.03 per px and worth ~6 MHz. It changes with alignment (the 9-2 and 9-7
states differ by 9-12 % on every line), so it is measured per scan, from
the scan's own calibration frames, like everything else in the chain
(PI decision 2026-09-09: envelope_source = "measured" stays; a stored
ePSF table carries its sweep's envelope as provenance only).

Method (validated 2026-09-06, Data/2026-9-3/envelope_from_calibration.py):
each calibration frame carries four lines,

    outer_left  = FSR - f   (-f sideband, order 1)
    inner_left  = FSR + f   (+f sideband, order 1)
    inner_right = 2 FSR - f (-f sideband, order 2)
    outer_right = 2 FSR + f (+f sideband, order 2)

The same-sideband pairs (inner_left, outer_right) and (outer_left,
inner_right) have the SAME drive amplitude, so the log ratio of their areas
is g(x_a) - g(x_b) with the EOM drive roll-off and the +f/-f asymmetry
cancelled. Areas are summed counts within +-AREA_HALF_PX of each line minus
a local background (no lineshape fit). Over the sweep the pairs sample
g(x_a) - g(x_b) for many (x_a, x_b), and g(x) is fitted as a polynomial of
degree DEG in x (the constant is unidentifiable and irrelevant). The slope
g'(x) at a sample peak's position is what the fit applies.

One class, `Envelope`, owns the measurement (the pair samples), the fit and
the outputs the chain reads (`slope`, `ln_envelope`, `__call__`, `factor`),
plus save/load as a CSV and `compare` between two envelopes — the same
shape as `Epsf` for the kernel (2026-09-09). `Envelope.from_calibration`
does the image -> spectrum step; `envelope_from_calibration` is the thin
wrapper the fitter chain calls.

Measured 2026-09-06 on four 41-point calibrations of one session: inner
slopes +0.0095..+0.0098 / -0.0209..-0.0218 per px, fit rms 0.03-0.04 in ln
units — stable to 3 % scan to scan within an alignment state. 2026-09-09
(Data/2026-9-9/envelope_compare.txt): 401-point fine sweeps and 41-point
calibrations of the same day agree to 1-2 % on the inner slopes; within a
day the slopes scatter 2 % (inner, outer_left) and 6 % (outer_right, the
end of the sampled range); 9-2 vs 9-7 alignment states differ 9-12 %.
"""
from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

AREA_HALF_PX = 6
DEG = 4
X_CENTRE, X_SCALE = 100.0, 50.0        # design-matrix conditioning
MIN_FRAMES = 12
AREA_METHODS = ("sum", "template")
LINE_NAMES = ("outer_left", "left", "right", "outer_right")
PAIRS = (("left", "outer_right"), ("outer_left", "right"))   # +f, -f
# Rule of thumb for the centre pull of a gradient across a water line:
# ~6 MHz per 0.03/px on an outer order (2026-09-06 synthetic, production
# window). compare() multiplies a slope difference by this.
MHZ_PER_SLOPE = 200.0

_SAMPLE_DTYPE = np.dtype([("frame", "i4"), ("freq_ghz", "f8"), ("pair", "U2"),
                          ("x_a", "f8"), ("x_b", "f8"), ("ln_ratio", "f8"),
                          ("residual", "f8")])


def _u(x):
    return (np.asarray(x, dtype=float) - X_CENTRE) / X_SCALE


def _four_lines(px, y):
    idx, info = find_peaks(y, prominence=(y.max() - np.median(y)) * 0.015,
                           distance=6)
    if len(idx) < 4:
        return None
    order = np.argsort(info["prominences"])[::-1]
    inner = sorted(idx[order[:2]])
    lefts = [i for i in idx if i < inner[0] - 6]
    rights = [i for i in idx if i > inner[1] + 6]
    if not lefts or not rights:
        return None
    return [max(lefts, key=lambda i: y[i]), inner[0], inner[1],
            max(rights, key=lambda i: y[i])]


def _areas_sum(px, y, pk, half):
    """Summed counts within +-half px of each line minus the median of the
    pixels 7..16 px away that are clear of every line. None when a window
    leaves the ROI or the background has too few pixels."""
    far = np.ones_like(px, dtype=bool)
    for i in pk:
        far &= np.abs(px - px[i]) > half + 1
    out = []
    for i in pk:
        m = np.abs(px - px[i]) <= half
        b = far & (np.abs(px - px[i]) <= 16)
        if b.sum() < 3 or (px[i] - half) < px.min() \
                or (px[i] + half) > px.max():
            return None
        out.append(float(np.sum(y[m] - np.median(y[b]))))
    return out


def _areas_template(px, y, pk, half, profiles):
    """Area = amplitude of the line's unit-area ePSF profile (the node table
    blended at the line's position) fitted linearly with a local offset
    over +-half px; the centre is refined on a 0.05 px grid around the
    finder pixel. The template carries the wings, so the estimate does not
    depend on how much of them the window holds."""
    out = []
    for line, i in enumerate(pk):
        c0 = float(px[i])
        if (c0 - half) < px.min() or (c0 + half) > px.max():
            return None
        m = np.abs(px - c0) <= half
        best = None
        for c in c0 + np.arange(-1.0, 1.0 + 1e-9, 0.05):
            try:
                k = profiles.kernel(line, c)
            except ValueError:
                return None                       # beyond the node ladder
            t = np.interp(px[m] - c, k.u, k.k, left=0.0, right=0.0)
            M = np.column_stack([t, np.ones(t.size)])
            sol, *_ = np.linalg.lstsq(M, y[m], rcond=None)
            rss = float(np.sum((y[m] - M @ sol) ** 2))
            if best is None or rss < best[0]:
                best = (rss, float(sol[0]))
        out.append(best[1])
    return out


@dataclass(frozen=True)
class EnvelopeComparison:
    """Two envelopes at given positions: slopes [1/px], their difference and
    the predicted centre pull [MHz] (MHZ_PER_SLOPE rule of thumb)."""
    positions: np.ndarray
    slope_a: np.ndarray
    slope_b: np.ndarray
    d_slope: np.ndarray
    shift_mhz: np.ndarray

    def report(self, names=LINE_NAMES) -> str:
        lines = [f"{'position':>12s} {'x [px]':>7s} {'slope a':>9s} {'slope b':>9s} "
                 f"{'b - a':>9s} {'pull MHz':>9s}"]
        for j, x in enumerate(self.positions):
            nm = names[j] if j < len(names) else f"pos{j}"
            lines.append(f"{nm:>12s} {x:7.1f} {self.slope_a[j]:+9.4f} "
                         f"{self.slope_b[j]:+9.4f} {self.d_slope[j]:+9.4f} "
                         f"{self.shift_mhz[j]:+9.2f}")
        return "\n".join(lines)

    def __str__(self):
        return self.report()


class Envelope:
    """The ln VIPA envelope g(x) of one calibration sweep, from the
    same-sideband line-area pairs. See the module docstring.

    Attributes
    samples      structured array, one row per pair sample: frame, freq_ghz,
                 pair ('+f' | '-f'), x_a, x_b, ln_ratio = g(x_a) - g(x_b)
                 measured, residual after the fit
    coefficients c_1..c_deg of g(u) = sum c_k u^k, u = (x - 100) / 50
    n_frames     calibration frames that gave both pairs
    rms_ln       fit residual rms [ln]
    x_min, x_max coverage of the samples [px] (property `coverage`)
    deg, area_half_px, area_method   the fit and estimator settings
    """

    def __init__(self, freqs, pxs, slines, *, deg: int = DEG,
                 area_half_px: int = AREA_HALF_PX, area_method: str = "sum",
                 profiles=None, source: str = ""):
        """freqs        drive frequency per frame [GHz] (None = unknown)
        pxs, slines  pixel axis and row-summed spectrum per frame
        deg          polynomial degree of ln g(x)
        area_half_px half window of the area estimator [px]
        area_method  "sum" (counts minus a local median background) or
                     "template" (amplitude of the unit-area ePSF profile,
                     needs `profiles`, an Epsf of the same calibration)
        """
        if area_method not in AREA_METHODS:
            raise ValueError(f"area_method must be one of {AREA_METHODS}.")
        if area_method == "template" and profiles is None:
            raise ValueError("area_method = 'template' needs `profiles` (an Epsf).")
        self.deg = int(deg)
        self.area_half_px = int(area_half_px)
        self.area_method = area_method
        self.source = source
        self.path = None
        rows = []
        if freqs is None:
            freqs = [np.nan] * len(pxs)
        for k, (f, px, y) in enumerate(zip(freqs, pxs, slines)):
            px = np.asarray(px, dtype=float)
            y = np.asarray(y, dtype=float)
            pk = _four_lines(px, y)
            if pk is None:
                continue
            if area_method == "sum":
                A = _areas_sum(px, y, pk, self.area_half_px)
            else:
                A = _areas_template(px, y, pk, self.area_half_px, profiles)
            if A is None or min(A) <= 0:
                continue
            x_ol, x_il, x_ir, x_or = (px[i] for i in pk)
            rows.append((k, float(f), "+f", x_il, x_or, np.log(A[1] / A[3]), 0.0))
            rows.append((k, float(f), "-f", x_ol, x_ir, np.log(A[0] / A[2]), 0.0))
        if len(rows) < 2 * MIN_FRAMES:
            raise ValueError(
                f"Envelope: only {len(rows) // 2} calibration frames with four "
                f"usable lines (need {MIN_FRAMES}); is this a four-order ROI?")
        self.samples = np.array(rows, dtype=_SAMPLE_DTYPE)
        self._fit()

    # ----------------------------------------------------------- building
    @classmethod
    def from_calibration(cls, calibration_data, fitter, **kw) -> "Envelope":
        """The scan's envelope from its own calibration frames through the
        fitter's image -> spectrum step (four lines in the ROI required;
        raises ValueError otherwise so the caller can fall back)."""
        freqs, pxs, slines = [], [], []
        for block in calibration_data.measured_freqs:
            for point in block.cali_meas_points:
                px, y = fitter.get_px_sline_from_image(
                    np.asarray(point.frame, dtype=float))
                freqs.append(float(getattr(point, "microwave_freq", np.nan)))
                pxs.append(np.asarray(px, dtype=float))
                slines.append(np.asarray(y, dtype=float))
        return cls(freqs, pxs, slines, **kw)

    @classmethod
    def from_samples(cls, samples, *, deg: int = DEG,
                     area_half_px: int = AREA_HALF_PX, area_method: str = "sum",
                     source: str = "") -> "Envelope":
        """An Envelope refitted from stored pair samples (load, or a
        degree scan on the same measurement without re-reading frames)."""
        self = cls.__new__(cls)
        self.deg = int(deg)
        self.area_half_px = int(area_half_px)
        self.area_method = area_method
        self.source = source
        self.path = None
        self.samples = np.array(samples, dtype=_SAMPLE_DTYPE)
        self._fit()
        return self

    def _fit(self):
        s = self.samples
        M = self._design(s["x_a"], s["x_b"])
        c, *_ = np.linalg.lstsq(M, s["ln_ratio"], rcond=None)
        self.coefficients = np.asarray(c, dtype=float)
        resid = s["ln_ratio"] - M @ c
        s["residual"] = resid
        self.rms_ln = float(resid.std())
        self.n_frames = len(np.unique(s["frame"])) if s.size else 0
        self.x_min = float(min(s["x_a"].min(), s["x_b"].min()))
        self.x_max = float(max(s["x_a"].max(), s["x_b"].max()))

    def _design(self, xa, xb):
        return np.column_stack([_u(xa) ** j - _u(xb) ** j
                                for j in range(1, self.deg + 1)])

    def refit(self, deg: int) -> "Envelope":
        """The same samples fitted with another polynomial degree."""
        return Envelope.from_samples(self.samples, deg=deg,
                                     area_half_px=self.area_half_px,
                                     area_method=self.area_method,
                                     source=self.source)

    # ------------------------------------------------------------ outputs
    @property
    def coverage(self) -> tuple[float, float]:
        return (self.x_min, self.x_max)

    def ln_envelope(self, x):
        """g(x) = ln T(x), the log transmission, constant dropped (shape only)."""
        u = _u(x)
        return sum(c * u ** (k + 1) for k, c in enumerate(self.coefficients))

    def slope(self, x) -> float:
        """g'(x) = T'(x)/T(x) [1/px], the relative transmission slope the
        fits apply (a peak's amplitude absorbs T at its centre, only the
        tilt across the line is left)."""
        u = _u(x)
        return float(sum(c * (k + 1) * u ** k / X_SCALE
                         for k, c in enumerate(self.coefficients)))

    def transmission(self, x, x_ref: float = X_CENTRE):
        """The transmission T(x) / T(x_ref) = exp(g(x) - g(x_ref)), the
        envelope itself normalised at a reference pixel."""
        return np.exp(self.ln_envelope(x) - self.ln_envelope(x_ref))

    __call__ = transmission

    def ln_transmission(self, x):
        """g(x) = ln T(x), constant dropped (same as ln_envelope)."""
        return self.ln_envelope(x)

    def factor(self, x, c: float):
        """T(x) / T(c), the transmission over a fit window relative to the
        peak centre — the full-curve alternative to exp(g'(c) (x - c))."""
        return np.exp(self.ln_envelope(x) - self.ln_envelope(c))

    def residuals_vs_x(self, edges):
        """Mean and rms residual of the pair samples binned by the OUTER
        line's position (x_b of +f = outer_right, x_a of -f = outer_left;
        the inner positions move too little to bin on). Returns
        (centres, mean, rms, n) per bin."""
        s = self.samples
        x = np.where(s["pair"] == "+f", s["x_b"], s["x_a"])
        edges = np.asarray(edges, dtype=float)
        cen, mean, rms, n = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            cen.append(0.5 * (lo + hi))
            n.append(int(m.sum()))
            mean.append(float(s["residual"][m].mean()) if m.any() else np.nan)
            rms.append(float(np.sqrt(np.mean(s["residual"][m] ** 2))) if m.any() else np.nan)
        return np.array(cen), np.array(mean), np.array(rms), np.array(n)

    def compare(self, other: "Envelope", positions,
                mhz_per_slope: float = MHZ_PER_SLOPE) -> EnvelopeComparison:
        """Slope differences other - self at `positions` [1/px] and the
        predicted centre pull [MHz] from the MHZ_PER_SLOPE rule of thumb."""
        positions = np.asarray(positions, dtype=float)
        a = np.array([self.slope(x) for x in positions])
        b = np.array([other.slope(x) for x in positions])
        return EnvelopeComparison(positions=positions, slope_a=a, slope_b=b,
                                  d_slope=b - a, shift_mhz=(b - a) * mhz_per_slope)

    def __repr__(self):
        return (f"Envelope(deg={self.deg}, n_frames={self.n_frames}, "
                f"rms_ln={self.rms_ln:.3f}, coverage={self.x_min:.1f}-{self.x_max:.1f} px, "
                f"area={self.area_method} +-{self.area_half_px} px)")

    # ------------------------------------------------------------ storage
    def save(self, path, source: str = ""):
        """The pair samples to a CSV (one row per sample: frame, freq_ghz,
        pair, x_a, x_b, ln_ratio, residual) with the fit in comment lines
        (coefficients, degree, rms, coverage, estimator settings,
        provenance). Readable with pandas (comment="#")."""
        path = Path(str(path))
        source = source or self.source
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write(f"# VIPA ln-envelope pair samples (brillouin_system Envelope.save) "
                     f"written {_dt.datetime.now().isoformat(timespec='seconds')}; "
                     f"source={source or '-'}\n")
            fh.write(f"# ln g(u) = sum_k c_k u^k, u = (x - {X_CENTRE:g}) / {X_SCALE:g}; "
                     f"deg={self.deg}; coefficients="
                     + ",".join(f"{c:.10g}" for c in self.coefficients) + "\n")
            fh.write(f"# n_frames={self.n_frames}; rms_ln={self.rms_ln:.6g}; "
                     f"x_min={self.x_min:.4f}; x_max={self.x_max:.4f}; "
                     f"area_method={self.area_method}; area_half_px={self.area_half_px}\n")
            w = csv.writer(fh)
            w.writerow(["frame", "freq_ghz", "pair", "x_a", "x_b", "ln_ratio", "residual"])
            for r in self.samples:
                w.writerow([int(r["frame"]), f"{r['freq_ghz']:.6g}", str(r["pair"]),
                            f"{r['x_a']:.4f}", f"{r['x_b']:.4f}",
                            f"{r['ln_ratio']:.8g}", f"{r['residual']:.8g}"])
        self.path = str(path)

    @classmethod
    def load(cls, path) -> "Envelope":
        """An Envelope from a saved CSV: the samples are refitted with the
        stored degree and must reproduce the stored coefficients."""
        path = Path(str(path))
        meta = {}
        rows = []
        with path.open("r", newline="", encoding="utf-8") as fh:
            header = None
            for line in fh:
                if line.startswith("#"):
                    for part in line[1:].split(";"):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            meta[k.strip().split()[-1]] = v.strip()
                    continue
                if header is None:
                    header = next(csv.reader([line]))
                    continue
                if line.strip():
                    rows.append(next(csv.reader([line])))
        if header is None:
            raise ValueError(f"{path}: no sample table found.")
        col = {k: i for i, k in enumerate(header)}
        samples = [(int(r[col["frame"]]), float(r[col["freq_ghz"]]), r[col["pair"]],
                    float(r[col["x_a"]]), float(r[col["x_b"]]),
                    float(r[col["ln_ratio"]]), 0.0) for r in rows]
        self = cls.from_samples(samples, deg=int(meta.get("deg", DEG)),
                                area_half_px=int(meta.get("area_half_px", AREA_HALF_PX)),
                                area_method=meta.get("area_method", "sum"),
                                source=meta.get("source", ""))
        stored = meta.get("coefficients")
        if stored:
            c = np.array([float(v) for v in stored.split(",")])
            if c.size != self.coefficients.size or not np.allclose(c, self.coefficients, atol=1e-6):
                raise ValueError(f"{path}: stored coefficients do not match the refit.")
        self.path = str(path)
        return self


    @classmethod
    def from_coefficients(cls, coefficients, n_frames: int = 0,
                          rms_ln: float = float("nan"), x_min: float = float("nan"),
                          x_max: float = float("nan"), source: str = "") -> "Envelope":
        """An Envelope from fitted coefficients only (no samples) — the
        shape of the former EnvelopeModel dataclass, for diagnostics that
        construct a curve by hand (Data/2026-9-8/envelope_deck_figs.py)."""
        self = cls.__new__(cls)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.deg = int(self.coefficients.size)
        self.area_half_px = AREA_HALF_PX
        self.area_method = "sum"
        self.source = source
        self.path = None
        self.samples = np.zeros(0, dtype=_SAMPLE_DTYPE)
        self.n_frames = int(n_frames)
        self.rms_ln = float(rms_ln)
        self.x_min, self.x_max = float(x_min), float(x_max)
        return self


def EnvelopeModel(coefficients, n_frames=0, rms_ln=float("nan"),
                  x_min=float("nan"), x_max=float("nan")) -> Envelope:
    """Former name and constructor (2026-09-06 frozen dataclass); kept so
    the Data scripts that build one from coefficients still run."""
    return Envelope.from_coefficients(coefficients, n_frames, rms_ln, x_min, x_max)


def _areas(px, y, pk, half: int = AREA_HALF_PX):
    """Former module function (Data/2026-9-8 deck figures)."""
    return _areas_sum(px, y, pk, half)


def envelope_from_calibration(calibration_data, fitter, **kw) -> Envelope:
    """The scan's envelope from its own calibration frames (four lines in
    the ROI required; raises otherwise so the caller can fall back)."""
    return Envelope.from_calibration(calibration_data, fitter, **kw)
