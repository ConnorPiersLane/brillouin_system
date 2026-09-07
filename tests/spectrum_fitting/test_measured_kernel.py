"""Tests for the measured instrument kernel (spectrum_fitting/measured_kernel.py).

Synthetic elastic lines generated with the PARAMETRIC kernel are pushed
through the extraction; the measured kernel must then reproduce that
parametric kernel (an extraction control), and a DHO spectrum generated with
the parametric kernel must be recovered by a DHO fit that uses the measured
kernel — width and resonance alike.
"""
from dataclasses import dataclass, replace

import numpy as np
import pytest

from brillouin_system.spectrum_fitting.dho import DhoAxes, dho_profile
from brillouin_system.spectrum_fitting.measured_kernel import (
    MeasuredKernel,
    WINDOW_PX,
    build_measured_kernel,
    measured_kernels_for_frame,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
    SampleFindPeaksConfig,
)
from brillouin_system.spectrum_fitting.psf import DX, psf_profile
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

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
    f.update_sline_config(replace(
        f.sline_config, n_peaks=2, row_selection="manual",
        psf_sigma_left_px=SIGMA[0], psf_sigma_right_px=SIGMA[1],
        psf_tau_left_px=TAU[0], psf_tau_right_px=TAU[1]))
    f.update_reference_config(FindPeaksConfig(
        prominence_fraction=0.05, min_peak_width=1, min_peak_height=800,
        rel_height=0.5, wlen_pixels=40, fitting_model="lorentzian_x_psf",
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
            + psf_profile(PX, amp, cen_left, G_INST, SIGMA[0], TAU[0])
            + psf_profile(PX, amp, cen_right, G_INST, SIGMA[1], TAU[1]))


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
    grid = np.arange(-6.0, 6.0 + DX / 2, DX)
    k = psf_profile(grid, 1.0, 0.0, G_INST, SIGMA[idx], TAU[idx])
    return grid, k / (k.sum() * DX)


def test_extraction_reproduces_the_parametric_kernel():
    rng = np.random.default_rng(3)
    cal = synthetic_calibration(rng)
    fitter = make_fitter()
    for idx, c0 in ((0, CEN_LEFT), (1, CEN_RIGHT)):
        mk = build_measured_kernel(cal, fitter, c0, idx)
        assert isinstance(mk, MeasuredKernel)
        assert mk.n_frames >= 8
        assert abs(mk.g_median_px - G_INST) < 0.03
        grid, k_par = parametric_kernel(idx)
        assert np.allclose(mk.u, grid)
        assert abs(mk.k.sum() * DX - 1.0) < 1e-6
        # core and near wing to a few percent of the peak, far wing to 15 %
        peak = k_par.max()
        core = np.abs(grid) <= 1.5
        assert np.max(np.abs(mk.k[core] - k_par[core])) < 0.04 * peak
        for u in (2.0, 3.0, 4.0):
            j = int(np.argmin(np.abs(grid - u)))
            jm = int(np.argmin(np.abs(grid + u)))
            ratio = (mk.k[j] + mk.k[jm]) / (k_par[j] + k_par[jm])
            assert 0.85 < ratio < 1.15, (u, ratio)


def test_dho_fit_with_measured_kernel_recovers_width_and_resonance():
    rng = np.random.default_rng(4)
    cal = synthetic_calibration(rng)
    fitter = make_fitter()
    axes_par = DhoAxes(
        freq_left_poly=POLY_LEFT, freq_right_poly=POLY_RIGHT,
        instrument_width_left_poly=np.array([G_INST]),
        instrument_width_right_poly=np.array([G_INST]))
    gamma_ghz = 0.143                       # water at 22 C
    gam_px = (gamma_ghz / SLOPE_LEFT, gamma_ghz / abs(SLOPE_RIGHT))
    amp = (3400.0, 3400.0)
    # each core truncated to +-15 px: the even DHO core mirrors at nu = -nu_B
    # (px 44 on this linear track), which the windowed production fit never
    # sees but a synthetic full-frame spectrum must not contain
    reach = lambda c: np.abs(PX - c) <= 15.0
    sline = (FLOOR
             + dho_profile(PX, amp[0], CEN_LEFT, gam_px[0], POLY_LEFT, G_INST, SIGMA[0], TAU[0]) * reach(CEN_LEFT)
             + dho_profile(PX, amp[1], CEN_RIGHT, gam_px[1], POLY_RIGHT, G_INST, SIGMA[1], TAU[1]) * reach(CEN_RIGHT))
    # the sample line is left noise-free: a single noisy frame carries ~7 %
    # width noise, which would swamp the 3 % this test is about; the
    # calibration frames ARE noisy, so the extraction is tested under noise
    frame = frame_from_sline(sline)

    k_left, k_right = measured_kernels_for_frame(cal, fitter, frame)
    axes = replace(axes_par, kernel_left=k_left, kernel_right=k_right)
    assert axes.has_measured_kernels

    px, s = fitter.get_px_sline_from_image(frame)
    fit = fitter.fit(np.asarray(px, float), np.asarray(s, float),
                     is_reference_mode=False, dho_axes=axes)
    assert fit.is_success
    assert abs(fit.left_peak_center_px - CEN_LEFT) < 0.03
    assert abs(fit.right_peak_center_px - CEN_RIGHT) < 0.03
    assert abs(fit.left_peak_width_px / gam_px[0] - 1.0) < 0.03
    assert abs(fit.right_peak_width_px / gam_px[1] - 1.0) < 0.03

    # and the parametric chain on the same frame agrees (same truth)
    fit_par = fitter.fit(np.asarray(px, float), np.asarray(s, float),
                         is_reference_mode=False, dho_axes=axes_par)
    assert abs(fit_par.left_peak_width_px / fit.left_peak_width_px - 1.0) < 0.03
    assert abs(fit_par.left_peak_center_px - fit.left_peak_center_px) < 0.02


def test_measured_kernel_needs_frames_near_the_position():
    rng = np.random.default_rng(5)
    cal = synthetic_calibration(rng)
    with pytest.raises(ValueError, match="calibration frames"):
        build_measured_kernel(cal, make_fitter(), CEN_LEFT + 10 * WINDOW_PX, 0)


def test_config_validates_dho_kernel():
    cfg = SampleFindPeaksConfig(
        prominence_fraction=0.001, min_peak_width=1, min_peak_height=5,
        rel_height=0.5, wlen_pixels=10, fitting_model="dho_x_psf",
        dho_kernel="measured")
    assert cfg.dho_kernel == "measured"
    with pytest.raises(ValueError, match="dho_kernel"):
        SampleFindPeaksConfig(
            prominence_fraction=0.001, min_peak_width=1, min_peak_height=5,
            rel_height=0.5, wlen_pixels=10, fitting_model="dho_x_psf",
            dho_kernel="airy")
