"""Synthetic asymmetric spectral lines for the tests.

The production chain measures the instrument response (spectrum_fitting/
epsf.py) and carries no line model of its own, so the tests synthesise
lines with a small physical model of their own: a Lorentzian core, a
Gaussian blur, a one-sided exponential readout tail toward higher pixels
and the 1 px pixel box, convolved numerically on the fine grid. That is
what an elastic calibration line looks like on the detector, and what the
measured kernel must reproduce.
"""
from dataclasses import dataclass, replace

import numpy as np

from brillouin_system.spectrum_fitting.measured_kernel import DX, MeasuredKernel
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig, SampleFindPeaksConfig)
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

HALF_PX = 12.0
GRID = np.arange(-6.0, 6.0 + DX / 2, DX)


def camera_kernel(sigma, tau, box=0.0, half_px=4.0):
    """(x, k): the detector part alone, Gauss(sigma) x one-sided tail(tau)
    toward +x x 1 px pixel box [x optional row-tilt top-hat], unit sum on
    the DX grid. Compact, so a whole-axis line convolves cheaply."""
    n = int(round(half_px / DX))
    x = DX * (np.arange(2 * n + 1) - n)
    k = np.zeros_like(x)
    k[n] = 1.0
    if sigma > 0:
        g = np.exp(-0.5 * (x / sigma) ** 2)
        k = np.convolve(k, g / g.sum(), mode="same")
    if tau > 0:
        t = np.where(x >= 0, np.exp(-x / tau), 0.0)
        k = np.convolve(k, t / t.sum(), mode="same")
    # the 1 px pixel box as a symmetric trapezoid (odd length, half-weight
    # ends), i.e. an exact-to-O(DX^2) integral over [-0.5, +0.5] px
    pix = np.ones(int(round(1.0 / DX)) + 1)
    pix[0] = pix[-1] = 0.5
    k = np.convolve(k, pix / pix.sum(), mode="same")
    if box > 0:
        b = np.ones(max(int(round(box / DX)), 1))
        k = np.convolve(k, b / b.sum(), mode="same")
    return x, k / k.sum()


def instrument_kernel(gamma, sigma, tau, half_px=HALF_PX):
    """(x, k): unit-area Lorentzian(gamma) x camera kernel on the DX grid,
    x the offset of each sample from the Lorentzian centre (the core sits
    at zero offset, the convention the template centre differs from by a
    constant)."""
    n = int(round(half_px / DX))
    x = DX * (np.arange(2 * n + 1) - n)
    lor = 1.0 / (1.0 + (x / max(float(gamma), 1e-6)) ** 2)
    _, cam = camera_kernel(sigma, tau)
    k = np.clip(np.convolve(lor, cam, mode="same"), 0.0, None)
    return x, k / (k.sum() * DX)


def asym_line(px, amp, cen, gamma, sigma, tau, box=0.0):
    """One line of peak height `amp` at `cen` on pixels `px`: the analytic
    Lorentzian over the whole axis (its wings matter for the baseline)
    through the camera kernel. `box` adds a top-hat smear (row tilt)."""
    px = np.asarray(px, dtype=float)
    lo = float(px.min()) - float(cen) - 8.0
    hi = float(px.max()) - float(cen) + 8.0
    xf = lo + DX * np.arange(int(round((hi - lo) / DX)) + 1)
    lor = 1.0 / (1.0 + (xf / max(float(gamma), 1e-6)) ** 2)
    _, cam = camera_kernel(sigma, tau, box)
    y = np.convolve(lor, cam, mode="same")
    y = y / y.max()
    return float(amp) * np.interp(px - float(cen), xf, y)


def unit_kernel(gamma, sigma, tau, grid=GRID):
    """The unit-area kernel on the +-6 px GRID (a MeasuredKernel's k)."""
    x, k = instrument_kernel(gamma, sigma, tau)
    kk = np.interp(grid, x, k, left=0.0, right=0.0)
    return kk / (kk.sum() * DX)


def measured_kernel(gamma, sigma, tau, position_px=0.0, n_frames=14):
    k = unit_kernel(gamma, sigma, tau)
    return MeasuredKernel(u=GRID, k=k, position_px=float(position_px),
                          n_frames=n_frames, g_median_px=float(gamma))


# ---------------------------------------------------------------- a synthetic
# two-line calibration sweep on a 27-row frame (the sline sums rows 7..19)
SIGMA = (0.26, 0.27)
TAU = (0.39, 0.17)
G_INST = 0.43
CEN_LEFT, CEN_RIGHT = 80.0, 116.0
SLOPE_LEFT, SLOPE_RIGHT = 0.279, -0.350          # GHz/px
POLY_LEFT = np.array([SLOPE_LEFT, 5.07 - SLOPE_LEFT * CEN_LEFT])
POLY_RIGHT = np.array([SLOPE_RIGHT, 5.07 - SLOPE_RIGHT * CEN_RIGHT])
FLOOR = 2600.0
N_ROWS = 27
ROW_BAND = slice(7, 20)          # 13 rows, the sline band
PX = np.arange(200, dtype=float)


def make_fitter() -> SpectrumFitter:
    f = SpectrumFitter()
    f.update_sline_config(replace(f.sline_config, n_peaks=2, row_selection="manual",
                                  selected_rows=list(range(7, 20)), kernel_source="scan"))
    f.update_reference_config(FindPeaksConfig(
        prominence_fraction=0.05, min_peak_width=1, min_peak_height=800,
        rel_height=0.5, wlen_pixels=40, fitting_model="lorentzian",
        background="flat"))
    f.update_sample_config(SampleFindPeaksConfig(
        prominence_fraction=0.001, min_peak_width=1, min_peak_height=5,
        rel_height=0.5, wlen_pixels=10, fitting_model="dho_x_psf",
        background="flat"))
    return f


def frame_from_sline(sline: np.ndarray) -> np.ndarray:
    """A 2D frame whose row-band sum equals `sline`, plus a bias floor
    everywhere (the fitter sums ROW_BAND)."""
    frame = np.full((N_ROWS, PX.size), FLOOR / 13.0)
    frame[ROW_BAND, :] = (sline / 13.0)[None, :]
    return frame


def elastic_sline(cen_left, cen_right, amp=60000.0):
    return (FLOOR
            + asym_line(PX, amp, cen_left, G_INST, SIGMA[0], TAU[0])
            + asym_line(PX, amp, cen_right, G_INST, SIGMA[1], TAU[1]))


@dataclass
class _Point:
    frame: np.ndarray
    microwave_freq: float


@dataclass
class _Block:
    set_freq_ghz: float
    cali_meas_points: list


@dataclass
class _Calibration:
    measured_freqs: list


def synthetic_calibration(rng, n_points=41):
    """A 4-8 GHz EOM sweep: the left line walks +0.36 px/step, the right
    line -0.29 px/step (the real dispersions), noise as measured."""
    blocks = []
    for k in range(n_points):
        f = 4.0 + 0.1 * k
        cl = CEN_LEFT + (f - 5.07) / SLOPE_LEFT
        cr = CEN_RIGHT + (f - 5.07) / SLOPE_RIGHT
        s = elastic_sline(cl, cr)
        s = s + rng.normal(0.0, np.sqrt(s / 3.89 + 1.37))
        blocks.append(_Block(f, [_Point(frame_from_sline(s), f)]))
    return _Calibration(blocks)


def parametric_kernel(idx):
    """(grid, k) of the synthetic inner line idx (0 left, 1 right)."""
    return GRID, unit_kernel(G_INST, SIGMA[idx], TAU[idx])
