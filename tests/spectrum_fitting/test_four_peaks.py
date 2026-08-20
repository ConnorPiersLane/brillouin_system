"""Tests for the n_peaks config option: 2 = production inner pair,
4 = all VIPA orders, selection by amplitude ranking, per-position tails
(pr_tau_outer_* for the outer orders, measured 2026-08-20).
"""
import numpy as np
import pytest

from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
    PixelResponseConstants,
)
from brillouin_system.spectrum_fitting.pixel_response import pixel_response_profile
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

SIGMA, TAU_L, TAU_R = 0.25, 0.4, 0.2
TAU_OL, TAU_OR = 0.5, 0.0
# the real 4-peak ROI geometry: S_outer / AS / S / AS_outer
CENTERS = (39.0, 83.0, 118.0, 146.0)
TAUS = (TAU_OL, TAU_L, TAU_R, TAU_OR)
AMPS = (1800.0, 3000.0, 3000.0, 1800.0)   # outer ~60% of the main pair
GAMMA = 1.0
OFFSET = 80.0


def make_config(model="prm0", n_peaks=2) -> FindPeaksConfig:
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=50,
        rel_height=0.5,
        wlen_pixels=20,
        fitting_model=model,
        n_peaks=n_peaks,
    )


def make_fitter(model="prm0", n_peaks=2) -> SpectrumFitter:
    fitter = SpectrumFitter()
    fitter.pr_config = PixelResponseConstants(
        pr_sigma_px=SIGMA, pr_tau_left_px=TAU_L, pr_tau_right_px=TAU_R,
        pr_tau_outer_left_px=TAU_OL, pr_tau_outer_right_px=TAU_OR)
    fitter.update_sample_config(make_config(model, n_peaks))
    fitter.update_reference_config(make_config("pixel_response"))
    return fitter


def make_spectrum(seed=0):
    px = np.arange(0.0, 200.0)
    true = np.full_like(px, OFFSET)
    for a, c, tau in zip(AMPS, CENTERS, TAUS):
        true = true + pixel_response_profile(px, a, c, GAMMA, SIGMA, tau)
    rng = np.random.default_rng(seed)
    return px, true + rng.normal(0.0, 2.0, size=true.shape)


def test_n_peaks_validation():
    with pytest.raises(ValueError, match="n_peaks"):
        make_config(n_peaks=3)


def test_four_peaks_needs_supported_lineshape():
    fitter = make_fitter("voigt", n_peaks=4)
    # reference must match the sample family, or the model-mixing guard
    # fires first and masks the n_peaks error
    fitter.update_reference_config(make_config("voigt"))
    px, sline = make_spectrum()
    with pytest.raises(ValueError, match="n_peaks=4"):
        fitter.fit(px, sline, is_reference_mode=False)


def test_two_peak_default_picks_the_inner_pair():
    # amplitude ranking: the brightest two are the inner main pair
    fitter = make_fitter("prm0", n_peaks=2)
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    assert result.model.startswith("2")
    assert abs(result.left_peak_center_px - CENTERS[1]) < 0.05
    assert abs(result.right_peak_center_px - CENTERS[2]) < 0.05


def test_four_peak_fit_recovers_all_orders():
    fitter = make_fitter("prm0", n_peaks=4)
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    assert result.model.startswith("4")
    # all four centres in the parameter vector (left-to-right, 3 per peak)
    fitted_centers = [float(result.parameters[3 * i + 1]) for i in range(4)]
    for got, want in zip(fitted_centers, CENTERS):
        assert abs(got - want) < 0.05, (got, want)
    fitted_widths = [float(result.parameters[3 * i + 2]) for i in range(4)]
    for w in fitted_widths:
        assert abs(w - GAMMA) < 0.05
    # the REPORTED left/right stay the inner main pair
    assert abs(result.left_peak_center_px - CENTERS[1]) < 0.05
    assert abs(result.right_peak_center_px - CENTERS[2]) < 0.05
    assert abs(result.inter_peak_distance
               - (CENTERS[2] - CENTERS[1])) < 0.1


def test_four_peak_matches_two_peak_on_main_pair():
    # the main-pair centres must not move between the 2- and 4-peak fits
    px, sline = make_spectrum()
    r2 = make_fitter("prm0", n_peaks=2).fit(px, sline,
                                            is_reference_mode=False)
    r4 = make_fitter("prm0", n_peaks=4).fit(px, sline,
                                            is_reference_mode=False)
    assert abs(r2.left_peak_center_px - r4.left_peak_center_px) < 0.03
    assert abs(r2.right_peak_center_px - r4.right_peak_center_px) < 0.03


def test_four_peak_wrong_outer_tau_biases_outer_centre():
    # sanity that the per-position tails matter: fitting with the outer
    # tails swapped moves the outer centres by the tail convention
    fitter = make_fitter("prm0", n_peaks=4)
    fitter.pr_config = PixelResponseConstants(
        pr_sigma_px=SIGMA, pr_tau_left_px=TAU_L, pr_tau_right_px=TAU_R,
        pr_tau_outer_left_px=TAU_OR, pr_tau_outer_right_px=TAU_OL)
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    c_outer_left = float(result.parameters[1])
    assert abs(c_outer_left - CENTERS[0]) > 0.15   # visibly biased
