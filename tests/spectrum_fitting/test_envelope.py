"""Tests for the per-scan VIPA envelope (spectrum_fitting/envelope.py).

Synthetic four-line calibration frames with a known ln-envelope g(x): the
same-sideband area ratios must recover g'(x) at the inner-line positions.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from brillouin_system.spectrum_fitting.envelope import (
    EnvelopeModel, envelope_from_calibration)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    SlineFromFrameConfig)
from brillouin_system.spectrum_fitting.psf import psf_profile
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter
from dataclasses import replace

PX = np.arange(200, dtype=float)
FLOOR = 2600.0
N_ROWS, ROW_BAND = 27, slice(7, 20)
# real geometry: FSR 21.7 GHz; tracks px(f) for the four lines
FSR = 21.7
def x_outer_left(f): return 37.0 - (f - 5.07) / 0.197
def x_left(f): return 80.0 + (f - 5.07) / 0.279
def x_right(f): return 116.0 - (f - 5.07) / 0.350
def x_outer_right(f): return 143.5 + (f - 5.07) / 0.403
# true ln envelope: a parabola peaking at px 94, with a linear term
def g_true(x): return -0.0003 * (x - 94.0) ** 2 + 0.002 * (x - 94.0)
def gprime_true(x): return -0.0006 * (x - 94.0) + 0.002


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


def frame_from_sline(sline):
    frame = np.full((N_ROWS, PX.size), FLOOR / 13.0)
    frame[ROW_BAND, :] = (sline / 13.0)[None, :]
    return frame


def synthetic_calibration(rng, drive_rolloff=True):
    """Four lines per frame; the +f pair (left, outer_right) share one drive
    amplitude, the -f pair (outer_left, right) another; both roll off with
    f; every line is multiplied by the envelope exp(g(x))."""
    blocks = []
    for k in range(41):
        f = 4.0 + 0.1 * k
        a_plus = 60000.0 * (1.0 - 0.08 * (f - 4.0)) if drive_rolloff else 60000.0
        a_minus = 0.85 * a_plus
        s = np.full(PX.size, FLOOR)
        for x0, amp in ((x_outer_left(f), a_minus), (x_left(f), a_plus),
                        (x_right(f), a_minus), (x_outer_right(f), a_plus)):
            s = s + amp * np.exp(g_true(x0)) * psf_profile(PX, 1.0, x0, 0.43, 0.26, 0.3)
        s = s + rng.normal(0.0, np.sqrt(s / 3.89 + 1.37))
        blocks.append(_Block(f, [_Point(frame_from_sline(s), f)]))
    return _Calibration(blocks)


def make_fitter():
    f = SpectrumFitter()
    f.update_sline_config(replace(f.sline_config, n_peaks=4, row_selection="manual"))
    return f


def test_envelope_slopes_recovered_from_area_ratios():
    rng = np.random.default_rng(7)
    env = envelope_from_calibration(synthetic_calibration(rng), make_fitter())
    assert isinstance(env, EnvelopeModel)
    assert env.n_frames >= 30
    assert env.rms_ln < 0.06
    for x in (80.0, 116.0, 37.0, 143.5):
        assert abs(env.slope(x) - gprime_true(x)) < 0.003, (x, env.slope(x), gprime_true(x))


def test_drive_rolloff_cancels():
    rng = np.random.default_rng(8)
    with_roll = envelope_from_calibration(synthetic_calibration(rng, True), make_fitter())
    without = envelope_from_calibration(synthetic_calibration(rng, False), make_fitter())
    for x in (80.0, 116.0):
        assert abs(with_roll.slope(x) - without.slope(x)) < 0.002


def test_config_validates_envelope_source():
    cfg = SlineFromFrameConfig(pixel_offset_left=0, pixel_offset_right=0,
                               selected_rows=list(range(7, 20)), envelope_source="measured")
    assert cfg.envelope_source == "measured"
    with pytest.raises(ValueError, match="envelope_source"):
        SlineFromFrameConfig(pixel_offset_left=0, pixel_offset_right=0,
                             selected_rows=list(range(7, 20)), envelope_source="guess")
