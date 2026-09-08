"""Per-scan VIPA envelope from the calibration sweep — model-free.

The VIPA transmission varies across the detector (the envelope). It
multiplies every line by exp(g(x)), so a line of finite width is tilted by
the local slope g'(x). For the inner pair that slope is ~0.01-0.02 per px
and moves the fitted DHO resonance by ~2 MHz; for the outer orders it is
~0.03 per px and worth ~6 MHz. It changes with alignment (the 9-2 and 9-6
sweeps differ by a factor two on the inner pair), so it has to be measured
per scan, from the scan's own calibration frames, like everything else in
the chain.

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
g'(x) at a sample peak's position is what the fit needs.

Measured 2026-09-06 on four 41-point calibrations of one session: inner
slopes +0.0095..+0.0098 / -0.0209..-0.0218 per px, fit rms 0.03-0.04 in ln
units — stable to 3 % scan to scan within an alignment state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

AREA_HALF_PX = 6
DEG = 4
X_CENTRE, X_SCALE = 100.0, 50.0        # design-matrix conditioning
MIN_FRAMES = 12


@dataclass(frozen=True)
class EnvelopeModel:
    """ln envelope g(x) = sum_k c_k ((x - X_CENTRE)/X_SCALE)^k, k = 1..DEG."""
    coefficients: np.ndarray
    n_frames: int
    rms_ln: float
    x_min: float
    x_max: float

    def _u(self, x):
        return (np.asarray(x, dtype=float) - X_CENTRE) / X_SCALE

    def ln_envelope(self, x):
        u = self._u(x)
        return sum(c * u ** (k + 1) for k, c in enumerate(self.coefficients))

    def slope(self, x) -> float:
        """g'(x) [1/px] — the per-line envelope slope the fits apply."""
        u = self._u(x)
        return float(sum(c * (k + 1) * u ** k / X_SCALE
                         for k, c in enumerate(self.coefficients)))


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


def _areas(px, y, pk):
    far = np.ones_like(px, dtype=bool)
    for i in pk:
        far &= np.abs(px - px[i]) > AREA_HALF_PX + 1
    out = []
    for i in pk:
        m = np.abs(px - px[i]) <= AREA_HALF_PX
        b = far & (np.abs(px - px[i]) <= 16)
        if b.sum() < 3 or (px[i] - AREA_HALF_PX) < px.min() \
                or (px[i] + AREA_HALF_PX) > px.max():
            return None
        out.append(float(np.sum(y[m] - np.median(y[b]))))
    return out


def envelope_from_calibration(calibration_data, fitter) -> EnvelopeModel:
    """The scan's envelope from its own calibration frames (four lines in
    the ROI required; raises otherwise so the caller can fall back)."""
    rows = []
    for block in calibration_data.measured_freqs:
        for point in block.cali_meas_points:
            px, y = fitter.get_px_sline_from_image(np.asarray(point.frame, dtype=float))
            px = np.asarray(px, dtype=float)
            y = np.asarray(y, dtype=float)
            pk = _four_lines(px, y)
            if pk is None:
                continue
            A = _areas(px, y, pk)
            if A is None or min(A) <= 0:
                continue
            x_ol, x_il, x_ir, x_or = (px[i] for i in pk)
            rows.append((x_il, x_or, np.log(A[1] / A[3])))     # +f pair
            rows.append((x_ol, x_ir, np.log(A[0] / A[2])))     # -f pair
    if len(rows) < 2 * MIN_FRAMES:
        raise ValueError(
            f"Envelope: only {len(rows) // 2} calibration frames with four "
            f"usable lines (need {MIN_FRAMES}); is this a four-order ROI?")
    r = np.asarray(rows)
    xa, xb, dg = r[:, 0], r[:, 1], r[:, 2]
    u = lambda x: (x - X_CENTRE) / X_SCALE
    M = np.column_stack([u(xa) ** j - u(xb) ** j for j in range(1, DEG + 1)])
    c, *_ = np.linalg.lstsq(M, dg, rcond=None)
    resid = dg - M @ c
    return EnvelopeModel(coefficients=c, n_frames=len(rows) // 2,
                         rms_ln=float(resid.std()),
                         x_min=float(min(xa.min(), xb.min())),
                         x_max=float(max(xa.max(), xb.max())))
