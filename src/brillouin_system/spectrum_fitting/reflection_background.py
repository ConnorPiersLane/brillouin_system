"""The measured reflection-plane background ("ReflectionBG") and its mapper.

The laser carries a stable satellite comb (measured 2026-08-19: components at
~-0.9 / +-1.6 / +-4.0 / +-6.8 GHz from the carrier, pattern stable to 0.03%
and linear in intensity over a 5x range). Every elastic light path images this
comb through the VIPA, which puts a faint frequency-anchored background under
the Brillouin peaks — the structure prm1's linear slope used to absorb.

The production correction fits the measured pattern with ONE shared scale:

    lorentzian_x_psf pair + per-peak flat offset + s * R

(the 'prmr' preset / 'reflection' background in SpectrumFitter). A per-peak
scale was tested 2026-08-20 and rejected: freeing s on each side removes the
S-side constraint on the scale and re-opens the amplitude<->centre trade
(splits +3..+4 MHz on wide glycerol). An envelope change after realignment is
an instrument-state property — verify or retake the template, never free the
fit. This module supplies R:

* ReflectionBackground — a stored background measurement: the bias-subtracted
  mean 2D frame plus the (EOM freq, left px, right px) calibration points of
  its OWN session. Stored 2D so the row band can be re-selected at apply time.
* ReflectionBackgroundMapper — registers a ReflectionBackground onto another
  session's pixel axis through both calibrations.

Registration happens in FREQUENCY space, never as a pixel shift: the comb is
anchored to the laser line, so its position in GHz survives a VIPA
realignment while its position in pixels does not. The template sline is
read as a spectral density (counts per GHz) via the template session's
px<->GHz mapping and rendered back to pixels via the target session's
mapping, including both dispersion Jacobians. Consequences:

* A different ROI needs no bookkeeping: the 4-peak 200-px template renders
  onto an 85-px axis (or any crop) purely through the two calibrations.
* Alignment shifts AND dispersion changes are handled; what registration
  cannot fix is a change of the instrument lineshape itself — after a major
  realignment, retake the background (a jump of the in-window fit rms above
  its usual 10-26 counts is the alarm).
* No free shift parameter, on purpose: a fitted template shift trades against
  the AS peak centre at ~5 MHz/px (measured 2026-08-19), while calibration
  registration is good to ~0.1-0.2 px. The fit must never choose the shift.

Y alignment: the vertical line position also moves with alignment, so the
template stores the full 2D frame and its row band is selected from ITS OWN
frame (row_selection.select_rows) with the row count the sample sline uses —
the mirror of what SpectrumFitter does on the sample side. Any residual
row-capture difference is a pure scale and lands in the fitted s.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brillouin_system.spectrum_fitting.row_selection import select_rows

# Packaged default template (built by build_reflection_background.py).
REFLECTION_BG_DATA_DIR = Path(__file__).parent / "reflection_background_data"
DEFAULT_REFLECTION_BG = REFLECTION_BG_DATA_DIR / "reflection_bg_2026-08-19_4pk.npz"

# Track validity beyond the swept EOM range: the 4-8 GHz sweeps tolerate this
# much extrapolation of the quadratic tracks (the run_bg19 3.3-8.7 GHz window).
G_MARGIN_GHZ = 0.7

LEFT, RIGHT = 0, 1


@dataclass
class ReflectionBackground:
    """A stored reflection-plane background measurement.

    frame        bias-subtracted mean 2D frame (counts), full acquisition ROI.
    cal_freqs    EOM frequencies of the session's own calibration [GHz].
    cal_left_px  left(AS)-order sideband pixel position at each frequency.
    cal_right_px right(S)-order sideband pixel position at each frequency.
    meta         provenance (source files, bias, scan count, dates, ...).
    """

    frame: np.ndarray
    cal_freqs: np.ndarray
    cal_left_px: np.ndarray
    cal_right_px: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.frame = np.asarray(self.frame, dtype=float)
        self.cal_freqs = np.asarray(self.cal_freqs, dtype=float)
        self.cal_left_px = np.asarray(self.cal_left_px, dtype=float)
        self.cal_right_px = np.asarray(self.cal_right_px, dtype=float)
        if self.frame.ndim != 2:
            raise ValueError("ReflectionBackground.frame must be a 2D image.")
        n = len(self.cal_freqs)
        if not (len(self.cal_left_px) == len(self.cal_right_px) == n) or n < 3:
            raise ValueError(
                "ReflectionBackground needs >= 3 calibration points with "
                "matching freqs / left px / right px arrays."
            )
        self._slines: dict = {}
        self._px_of_g: dict = {}

    # ---------------- template axes ----------------

    @property
    def px(self) -> np.ndarray:
        """The template's own pixel axis (column indices of its frame)."""
        return np.arange(self.frame.shape[1], dtype=float)

    @property
    def g_lo(self) -> float:
        return float(np.min(self.cal_freqs)) - G_MARGIN_GHZ

    @property
    def g_hi(self) -> float:
        return float(np.max(self.cal_freqs)) + G_MARGIN_GHZ

    def px_of_g(self, order: int) -> np.ndarray:
        """Quadratic g[GHz] -> template px for one order (LEFT=0, RIGHT=1)."""
        if order not in self._px_of_g:
            x = self.cal_left_px if order == LEFT else self.cal_right_px
            self._px_of_g[order] = np.polyfit(self.cal_freqs, x, 2)
        return self._px_of_g[order]

    # ---------------- y alignment ----------------

    def sline(self, n_rows: int | None = None) -> np.ndarray:
        """Row-band sum of the template frame.

        The band is selected on the template's OWN frame (the pattern's own
        vertical position), with the same row count the sample sline uses.
        n_rows=None sums all rows.
        """
        key = int(n_rows) if n_rows is not None else None
        if key not in self._slines:
            if key is None:
                self._slines[key] = self.frame.sum(axis=0)
            else:
                rows = select_rows(self.frame, key)
                self._slines[key] = self.frame[rows, :].sum(axis=0)
        return self._slines[key]

    # ---------------- persistence ----------------

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame=self.frame,
            cal_freqs=self.cal_freqs,
            cal_left_px=self.cal_left_px,
            cal_right_px=self.cal_right_px,
            meta_json=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "ReflectionBackground":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                frame=z["frame"],
                cal_freqs=z["cal_freqs"],
                cal_left_px=z["cal_left_px"],
                cal_right_px=z["cal_right_px"],
                meta=json.loads(str(z["meta_json"])),
            )

    @classmethod
    def load_default(cls) -> "ReflectionBackground":
        """The packaged production template (2026-08-19, 4-peak ROI)."""
        if not DEFAULT_REFLECTION_BG.exists():
            raise FileNotFoundError(
                f"No packaged reflection background at "
                f"{DEFAULT_REFLECTION_BG}. Build one with "
                f"build_reflection_background.py from a reflection-plane "
                f"measurement."
            )
        return cls.load(DEFAULT_REFLECTION_BG)


def _freq_polys(calibration) -> tuple[np.ndarray, np.ndarray]:
    """The two px->GHz polynomials from whatever calibration form is given.

    Accepts a CalibrationCalculator, CalibrationPolyfitParameters, or a plain
    (freq_left_coeffs, freq_right_coeffs) pair.
    """
    if isinstance(calibration, (tuple, list)) and len(calibration) == 2:
        left, right = calibration
    else:
        params = getattr(calibration, "p", calibration)
        left = getattr(params, "freq_left_peak", None)
        right = getattr(params, "freq_right_peak", None)
    if left is None or right is None:
        raise ValueError(
            "Calibration provides no freq_left_peak/freq_right_peak "
            "polynomials — cannot register the reflection background."
        )
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise ValueError(
            "Calibration freq polynomials contain non-finite coefficients."
        )
    return left, right


class ReflectionBackgroundMapper:
    """Maps a ReflectionBackground onto a session's pixel axis.

    Takes the session's calibration (CalibrationCalculator,
    CalibrationPolyfitParameters, or a (freq_left, freq_right) coefficient
    pair) and the row count of the session's sline. render(px) then returns
    the background on that pixel axis, ready to pass to SpectrumFitter.fit
    as reflection_background (background='reflection' / the 'prmr'
    preset).

    Each order is rendered through its own track and confined to its own side
    of the axis. The split pixel is where the two tracks report the same
    offset (g_left == g_right) — the symmetry point of the order pair —
    computed on the given axis, so no peak positions are needed.
    """

    def __init__(self, background: ReflectionBackground, calibration,
                 n_rows: int | None = None):
        self.background = background
        self.freq_left, self.freq_right = _freq_polys(calibration)
        self._sline = background.sline(n_rows)

    def _mid_px(self, px: np.ndarray) -> float:
        d = np.polyval(self.freq_left, px) - np.polyval(self.freq_right, px)
        sign_change = np.nonzero(np.diff(np.signbit(d)))[0]
        if len(sign_change) == 0:
            raise ValueError(
                "The two calibration tracks never cross on this pixel axis — "
                "it does not cover a two-order ROI, so the reflection "
                "background cannot be split between the orders."
            )
        i = int(sign_change[0])
        # Linear interpolation of the zero crossing between the two pixels.
        return float(px[i] - d[i] * (px[i + 1] - px[i]) / (d[i + 1] - d[i]))

    def render(self, px) -> np.ndarray:
        """The reflection background on the given pixel axis [counts]."""
        px = np.asarray(px, dtype=float)
        mid = self._mid_px(px)
        bg = self.background
        out = np.zeros_like(px)
        sides = ((LEFT, self.freq_left, px <= mid),
                 (RIGHT, self.freq_right, px > mid))
        for order, fpoly, side in sides:
            g = np.polyval(fpoly, px)
            ok = side & (g >= bg.g_lo) & (g <= bg.g_hi) & np.isfinite(g)
            if not np.any(ok):
                continue
            c = bg.px_of_g(order)
            x_tmpl = np.polyval(c, g[ok])
            # counts/px -> counts/GHz on the template, back to counts/px here.
            dxdg = np.abs(np.polyval(np.polyder(c), g[ok]))
            dgdx = np.abs(np.polyval(np.polyder(fpoly), px[ok]))
            val = np.interp(x_tmpl, bg.px, self._sline, left=0.0, right=0.0)
            out[ok] = val * dxdg * dgdx
        return out
