"""Tests for the 'dho_x_psf' sample lineshape (spectrum_fitting/dho.py).

Geometry mirrors the real spectrometer: the left (anti-Stokes) peak's
frequency track RISES with px (its elastic line is to the LEFT), the right
(Stokes) track FALLS with px (elastic to the RIGHT). The eq.-S2 core is
mirror-symmetric about each peak's own elastic line with the heavy wing
toward LOWER SHIFT, so on the detector the leans point in OPPOSITE pixel
directions — left peak left, right peak right (the 2026-08-05 synthetic
result: absorbed slopes L negative, R positive).
"""
import numpy as np
import pytest

from dataclasses import replace

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationPolyfitParameters,
)
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.dho import DhoAxes, dho_profile
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FITTING_MODELS_REFERENCE,
    FITTING_MODELS_SAMPLE,
    FindPeaksConfig,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
    config_requires_dho_axes,
    is_dho_fit,
    is_psf_fit,
)

SIGMA = 0.25
TAU_LEFT = 0.4
TAU_RIGHT = 0.2

# px -> GHz shift from each peak's OWN elastic line. Left track rises with
# px (elastic left of the peak), right track falls (elastic right of it) —
# the real VIPA geometry. Resonances sit at 5.0 GHz on both tracks.
CEN_LEFT = 45.0
CEN_RIGHT = 90.0
SLOPE_LEFT = 0.32          # GHz/px
SLOPE_RIGHT = -0.36
POLY_LEFT = np.array([SLOPE_LEFT, 5.0 - SLOPE_LEFT * CEN_LEFT])
POLY_RIGHT = np.array([SLOPE_RIGHT, 5.0 - SLOPE_RIGHT * CEN_RIGHT])
G_INST_PX = 0.40           # instrument Lorentzian HWHM [px]
GAMMA_ACOUSTIC_GHZ = 0.35  # a viscous-fluid acoustic HWHM
AMP = 3000.0
OFFSET = 100.0

AXES = DhoAxes(
    freq_left_poly=POLY_LEFT,
    freq_right_poly=POLY_RIGHT,
    instrument_width_left_poly=np.array([G_INST_PX]),
    instrument_width_right_poly=np.array([G_INST_PX]),
)


def make_config(model: str) -> FindPeaksConfig:
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=50,
        rel_height=0.5,
        wlen_pixels=20,
        fitting_model=model,
    )


def make_fitter(sample_model="dho_x_psf",
                reference_model="lorentzian_x_psf") -> SpectrumFitter:
    fitter = SpectrumFitter()
    fitter.update_sample_config(make_config(sample_model))
    fitter.update_reference_config(make_config(reference_model))
    fitter.update_sline_config(replace(
        fitter.sline_config, n_peaks=2, psf_sigma_px=SIGMA,
        psf_tau_left_px=TAU_LEFT, psf_tau_right_px=TAU_RIGHT))
    return fitter


def gamma_px(slope):
    return GAMMA_ACOUSTIC_GHZ / abs(slope)


def make_spectrum():
    """Two noiseless DHO truths through the full instrument chain."""
    px = np.arange(16, 116, dtype=float)
    sline = (
        dho_profile(px, AMP, CEN_LEFT, gamma_px(SLOPE_LEFT), POLY_LEFT,
                    G_INST_PX, SIGMA, TAU_LEFT)
        + dho_profile(px, AMP, CEN_RIGHT, gamma_px(SLOPE_RIGHT), POLY_RIGHT,
                      G_INST_PX, SIGMA, TAU_RIGHT)
        + OFFSET
    )
    return px, sline


# ---------------- lean direction ----------------

def test_tail_directions_are_opposite_in_pixel_space():
    # Pure core (no camera blur or readout tail, negligible instrument
    # width): the wing toward LOWER SHIFT must be the heavier one — lower px
    # for the rising left track, higher px for the falling right track.
    px = np.arange(16, 116, dtype=float)
    d = 3.0
    left = dho_profile(px, 1.0, CEN_LEFT, gamma_px(SLOPE_LEFT), POLY_LEFT,
                       1e-6, 0.0, 0.0)
    il = int(np.argmin(np.abs(px - CEN_LEFT)))
    assert left[il - int(d)] > left[il + int(d)]

    right = dho_profile(px, 1.0, CEN_RIGHT, gamma_px(SLOPE_RIGHT), POLY_RIGHT,
                        1e-6, 0.0, 0.0)
    ir = int(np.argmin(np.abs(px - CEN_RIGHT)))
    assert right[ir + int(d)] > right[ir - int(d)]


def test_resonance_sits_above_the_apparent_maximum_in_shift():
    # Eq. S2's maximum is at nu^2 = nuB^2 - Gf^2/2, BELOW the resonance:
    # the apparent peak px must map to a smaller shift than polyval at cen.
    px = np.linspace(30.0, 60.0, 6001)
    prof = dho_profile(px, 1.0, CEN_LEFT, gamma_px(SLOPE_LEFT), POLY_LEFT,
                       1e-6, 0.0, 0.0)
    apparent = px[int(np.argmax(prof))]
    nu_apparent = np.polyval(POLY_LEFT, apparent)
    nu_res = np.polyval(POLY_LEFT, CEN_LEFT)
    assert nu_apparent < nu_res
    # and by roughly Gf^2/(4 nuB) (the box kernel shifts it only mildly)
    gf = 2.0 * GAMMA_ACOUSTIC_GHZ
    naive = gf ** 2 / (4.0 * nu_res)
    assert abs((nu_res - nu_apparent) - naive) < 0.6 * naive


# ---------------- closure ----------------

def test_dho_fit_recovers_resonance_and_acoustic_width():
    px, sline = make_spectrum()
    fitter = make_fitter()
    fit = fitter.fit(px, sline, is_reference_mode=False, dho_axes=AXES)
    assert fit.is_success
    assert is_dho_fit(fit.model)
    assert not is_psf_fit(fit.model)
    # noiseless self-closure: the resonance pixel comes back essentially
    # exactly, so the standard freq chain reports the true resonance
    assert abs(fit.left_peak_center_px - CEN_LEFT) < 0.02
    assert abs(fit.right_peak_center_px - CEN_RIGHT) < 0.02
    # fitted width is the ACOUSTIC HWHM in px
    assert abs(fit.left_peak_width_px - gamma_px(SLOPE_LEFT)) < 0.02
    assert abs(fit.right_peak_width_px - gamma_px(SLOPE_RIGHT)) < 0.02


def test_lorentzian_fit_of_dho_truth_is_pulled_toward_lower_shift():
    # The documented failure the DHO exists to fix: a symmetric fit of DHO
    # truth lands below the resonance in SHIFT — lower px on the rising
    # left track, higher px on the falling right track.
    px, sline = make_spectrum()
    fitter = make_fitter(sample_model="lorentzian_x_psf")
    fit = fitter.fit(px, sline, is_reference_mode=False)
    assert fit.is_success
    assert fit.left_peak_center_px < CEN_LEFT - 0.02
    assert fit.right_peak_center_px > CEN_RIGHT + 0.02


# ---------------- guards ----------------

def test_dho_refuses_reference_mode():
    px, sline = make_spectrum()
    fitter = make_fitter(reference_model="lorentzian_x_psf")
    fitter.update_reference_config(make_config("dho_x_psf"))
    with pytest.raises(ValueError, match="sample-only"):
        fitter.fit(px, sline, is_reference_mode=True, dho_axes=AXES)


def test_dho_requires_axes():
    px, sline = make_spectrum()
    fitter = make_fitter()
    with pytest.raises(ValueError, match="dho_axes"):
        fitter.fit(px, sline, is_reference_mode=False)


def test_dho_refuses_four_peaks():
    px, sline = make_spectrum()
    fitter = make_fitter()
    with pytest.raises(ValueError, match="n_peaks = 2 only"):
        fitter.fit(px, sline, is_reference_mode=False, n_peaks=4,
                   dho_axes=AXES)


def test_dho_single_found_peak_fails_instead_of_degrading():
    # A merged blob cannot be assigned to one elastic track.
    px = np.arange(16, 116, dtype=float)
    sline = dho_profile(px, AMP, CEN_LEFT, gamma_px(SLOPE_LEFT), POLY_LEFT,
                        G_INST_PX, SIGMA, TAU_LEFT) + OFFSET
    fitter = make_fitter()
    fit = fitter.fit(px, sline, is_reference_mode=False, dho_axes=AXES)
    assert not fit.is_success


def test_dho_sample_with_plain_lorentzian_reference_is_model_mixing():
    px, sline = make_spectrum()
    fitter = make_fitter(reference_model="lorentzian")
    with pytest.raises(ValueError, match="Model mixing"):
        fitter.fit(px, sline, is_reference_mode=False, dho_axes=AXES)


# ---------------- config wiring ----------------

def test_dho_is_a_sample_model_only():
    assert "dho_x_psf" in FITTING_MODELS_SAMPLE
    assert "dho_x_psf" not in FITTING_MODELS_REFERENCE
    cfg = make_config("dho_x_psf")
    assert cfg.fitting_model == "dho_x_psf"
    assert config_requires_dho_axes(cfg)
    assert not config_requires_dho_axes(make_config("lorentzian_x_psf"))


# ---------------- calibration side ----------------

def calibration_params(with_widths=True) -> CalibrationPolyfitParameters:
    p = CalibrationPolyfitParameters(
        degree=1,
        freq_left_peak=POLY_LEFT,
        freq_right_peak=POLY_RIGHT,
        freq_peak_distance=np.array([0.2, 0.0]),
    )
    if with_widths:
        p.calibration_width_left_peak = np.array([G_INST_PX])
        p.calibration_width_right_peak = np.array([G_INST_PX])
    return p


def test_calculator_builds_dho_axes():
    axes = CalibrationCalculator(calibration_params()).dho_axes()
    assert np.allclose(axes.freq_left_poly, POLY_LEFT)
    assert np.allclose(axes.instrument_width_right_poly, [G_INST_PX])


def test_calculator_without_width_model_refuses_dho_axes():
    calc = CalibrationCalculator(calibration_params(with_widths=False))
    with pytest.raises(ValueError, match="width"):
        calc.dho_axes()


def test_dho_linewidth_needs_no_instrument_subtraction():
    calc = CalibrationCalculator(calibration_params())
    fs = FittedSpectrum(
        is_success=True,
        model="2dho_x_psf_window",
        x_pixels=np.arange(16, 116, dtype=float),
        sline=np.zeros(100),
        left_peak_center_px=CEN_LEFT,
        left_peak_width_px=gamma_px(SLOPE_LEFT),
        right_peak_center_px=CEN_RIGHT,
        right_peak_width_px=gamma_px(SLOPE_RIGHT),
    )
    lw = calc.sample_linewidth_ghz(fs)
    hwhm = calc.hwhm_ghz(fs)
    assert lw == hwhm
    assert lw[0] == pytest.approx(GAMMA_ACOUSTIC_GHZ, rel=1e-6)
    assert lw[1] == pytest.approx(GAMMA_ACOUSTIC_GHZ, rel=1e-6)
    # the PSF path still subtracts (control): same numbers, psf tag
    fs_psf = FittedSpectrum(
        is_success=True,
        model="2lorentzian_x_psf_window",
        x_pixels=np.arange(16, 116, dtype=float),
        sline=np.zeros(100),
        left_peak_center_px=CEN_LEFT,
        left_peak_width_px=gamma_px(SLOPE_LEFT),
        right_peak_center_px=CEN_RIGHT,
        right_peak_width_px=gamma_px(SLOPE_RIGHT),
    )
    lw_psf = calc.sample_linewidth_ghz(fs_psf)
    assert lw_psf[0] < lw[0]
