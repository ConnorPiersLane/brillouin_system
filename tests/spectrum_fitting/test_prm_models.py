"""Wiring tests for the prm0/prm1 presets (pixel-response lineshape with a
per-peak flat / per-peak linear baseline), the flat_per_peak background and
the calibration/sample model-mixing guard.
"""
import numpy as np
import pytest

from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
    MODEL_PRESETS,
)
from brillouin_system.spectrum_fitting.pixel_response import pixel_response_profile
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
    model_requires_anchors,
    normalize_model_name,
)

SIGMA = 0.25
TAU_LEFT = 0.4
TAU_RIGHT = 0.2

CEN_LEFT = 25.0
CEN_RIGHT = 60.0
GAMMA = 1.0
AMP = 3000.0
OFF_LEFT = 100.0
OFF_RIGHT = 60.0


def make_config(model: str) -> FindPeaksConfig:
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=50,
        rel_height=0.5,
        wlen_pixels=20,
        fitting_model=model,
        pr_sigma_px=SIGMA,
        pr_tau_left_px=TAU_LEFT,
        pr_tau_right_px=TAU_RIGHT,
    )


def make_fitter(sample_model: str, reference_model: str) -> SpectrumFitter:
    fitter = SpectrumFitter()
    fitter.update_sample_config(make_config(sample_model))
    fitter.update_reference_config(make_config(reference_model))
    return fitter


def make_spectrum(seed=0):
    """Two pixel-response peaks on different per-peak pedestals."""
    px = np.arange(0, 86, dtype=float)
    mid = 0.5 * (CEN_LEFT + CEN_RIGHT)
    true = (
        pixel_response_profile(px, AMP, CEN_LEFT, GAMMA, SIGMA, TAU_LEFT)
        + pixel_response_profile(px, AMP, CEN_RIGHT, GAMMA, SIGMA, TAU_RIGHT)
        + np.where(px <= mid, OFF_LEFT, OFF_RIGHT)
    )
    rng = np.random.default_rng(seed)
    return px, true + rng.normal(0.0, 2.0, size=true.shape)


# ---------------- preset normalisation ----------------

def test_prm0_preset_expands():
    cfg = make_config("prm0")
    assert cfg.fitting_model == "pixel_response"
    assert cfg.background == "flat_per_peak"
    assert cfg.use_window is True
    assert cfg.beta == 3.0


def test_prm1_preset_expands():
    cfg = make_config("prm1")
    assert cfg.fitting_model == "pixel_response"
    assert cfg.background == "linear_per_peak"
    assert cfg.beta == 3.0


def test_preset_overrides_explicit_beta():
    cfg = make_config("prm1")
    # beta is pinned by the preset: the width recipe is only valid at 3.0.
    assert cfg.beta == 3.0
    assert "prm0" in MODEL_PRESETS and "prm1" in MODEL_PRESETS


def test_normalize_model_name_resolves_presets():
    assert normalize_model_name("prm0") == ("pixel_response", True)
    assert normalize_model_name("prm1") == ("pixel_response", True)
    assert not model_requires_anchors("prm0")
    assert not model_requires_anchors("prm1")


# ---------------- fitting ----------------

@pytest.mark.parametrize("model", ["prm0", "prm1"])
def test_prm_fits_recover_truth(model):
    fitter = make_fitter(model, model)
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    assert abs(result.left_peak_center_px - CEN_LEFT) < 0.05
    assert abs(result.right_peak_center_px - CEN_RIGHT) < 0.05
    assert abs(result.left_peak_width_px - GAMMA) < 0.05
    assert abs(result.right_peak_width_px - GAMMA) < 0.05
    assert "pixel_response" in result.model


def test_flat_per_peak_recovers_both_pedestals():
    fitter = make_fitter("prm0", "prm0")
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    # The reported offset is the mean background under the two peaks.
    assert abs(result.offset - 0.5 * (OFF_LEFT + OFF_RIGHT)) < 5.0


def test_direct_assignment_bypassing_post_init():
    # Scripts assign config.fitting_model after construction; the fitter must
    # still resolve the preset (background + beta included).
    fitter = make_fitter("lorentzian", "prm0")
    fitter.sample_config.fitting_model = "prm0"
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    assert "flat_per_peak" in result.model


# ---------------- the model-mixing guard ----------------

def test_mixing_pr_sample_with_lorentzian_reference_raises():
    fitter = make_fitter("prm1", "lorentzian")
    px, sline = make_spectrum()
    with pytest.raises(ValueError, match="[Mm]odel mixing"):
        fitter.fit(px, sline, is_reference_mode=False)


def test_mixing_lorentzian_sample_with_pr_reference_raises():
    fitter = make_fitter("lorentzian", "pixel_response")
    px, sline = make_spectrum()
    with pytest.raises(ValueError, match="[Mm]odel mixing"):
        fitter.fit(px, sline, is_reference_mode=False)


def test_mismatched_camera_constants_raise():
    fitter = make_fitter("prm1", "prm1")
    fitter.reference_config.pr_tau_left_px = 0.35  # differs from sample's 0.4
    px, sline = make_spectrum()
    with pytest.raises(ValueError, match="[Cc]amera-constant mismatch"):
        fitter.fit(px, sline, is_reference_mode=False)


def test_reference_mode_never_blocked():
    # The guard protects sample fits; calibrations fit standalone.
    fitter = make_fitter("lorentzian", "prm1")
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=True)
    assert result.is_success


def test_matched_lorentzian_pair_unaffected():
    fitter = make_fitter("lorentzian", "lorentzian")
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
