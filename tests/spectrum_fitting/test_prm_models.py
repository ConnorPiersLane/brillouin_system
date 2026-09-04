"""Wiring tests for the prm0/prm1 presets (pixel-response lineshape with a
flat / linear baseline — per-peak, since the presets fit windowed), the
legacy background-name normalisation and the calibration/sample
model-mixing guard.
"""
import numpy as np
import pytest

from dataclasses import replace

from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
    MODEL_PRESETS,
)
from brillouin_system.spectrum_fitting.psf import psf_profile
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
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
    )


def make_fitter(sample_model: str, reference_model: str) -> SpectrumFitter:
    fitter = SpectrumFitter()
    fitter.update_sample_config(make_config(sample_model))
    fitter.update_reference_config(make_config(reference_model))
    # camera kernel working values live in the [global] sline config; pin the
    # test values on the fitter directly
    fitter.update_sline_config(replace(
        fitter.sline_config, n_peaks=2,
        psf_sigma_left_px=SIGMA, psf_sigma_right_px=SIGMA,
        psf_tau_left_px=TAU_LEFT, psf_tau_right_px=TAU_RIGHT))
    return fitter


def make_spectrum(seed=0):
    """Two pixel-response peaks on different per-peak background offsets."""
    px = np.arange(0, 86, dtype=float)
    mid = 0.5 * (CEN_LEFT + CEN_RIGHT)
    true = (
        psf_profile(px, AMP, CEN_LEFT, GAMMA, SIGMA, TAU_LEFT)
        + psf_profile(px, AMP, CEN_RIGHT, GAMMA, SIGMA, TAU_RIGHT)
        + np.where(px <= mid, OFF_LEFT, OFF_RIGHT)
    )
    rng = np.random.default_rng(seed)
    return px, true + rng.normal(0.0, 2.0, size=true.shape)


# ---------------- preset normalisation ----------------

def test_prm0_preset_expands():
    cfg = make_config("prm0")
    assert cfg.fitting_model == "lorentzian_x_psf"
    assert cfg.background == "flat"
    assert cfg.use_window is True
    assert cfg.beta == 3.0


def test_prm1_preset_expands():
    cfg = make_config("prm1")
    assert cfg.fitting_model == "lorentzian_x_psf"
    assert cfg.background == "linear"
    assert cfg.beta == 3.0


def test_legacy_per_peak_background_names_normalise():
    # flat_per_peak/linear_per_peak were folded into flat/linear (2026-08-20):
    # the baseline scope follows use_window, so a windowed fit is per-peak
    # under the plain names too.
    cfg = FindPeaksConfig(
        prominence_fraction=0.05, min_peak_width=1, min_peak_height=50,
        rel_height=0.5, wlen_pixels=20, fitting_model="lorentzian_x_psf",
        background="linear_per_peak",
    )
    assert cfg.background == "linear"


def test_preset_overrides_explicit_beta():
    cfg = make_config("prm1")
    # beta is pinned by the preset: the width recipe is only valid at 3.0.
    assert cfg.beta == 3.0
    assert "prm0" in MODEL_PRESETS and "prm1" in MODEL_PRESETS


def test_normalize_model_name_resolves_presets():
    assert normalize_model_name("prm0") == ("lorentzian_x_psf", True)
    assert normalize_model_name("prm1") == ("lorentzian_x_psf", True)


def test_pixel_response_name_is_retired():
    # Renamed 2026-08-20, deliberately with NO alias: the old name raises
    # with the rename hint instead of quietly mapping.
    with pytest.raises(ValueError, match="renamed.*lorentzian_x_psf"):
        make_config("pixel_response")


def test_removed_models_raise_with_migration_hint():
    # voigt and the NA lineshape models were removed 2026-08-20; the config
    # refuses them with a hint instead of silently misbehaving.
    with pytest.raises(ValueError, match="removed"):
        make_config("voigt")
    with pytest.raises(ValueError, match="removed"):
        make_config("na_lorentzian")


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
    assert "lorentzian_x_psf" in result.model


def test_flat_per_peak_recovers_both_backgrounds():
    fitter = make_fitter("prm0", "prm0")
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    # The reported offset is the mean background under the two peaks.
    assert abs(result.offset - 0.5 * (OFF_LEFT + OFF_RIGHT)) < 5.0
    # And the per-peak levels travel with the fit (they feed the background-light
    # shot-noise term of the precision bound).
    assert abs(result.left_peak_bg_counts - OFF_LEFT) < 5.0
    assert abs(result.right_peak_bg_counts - OFF_RIGHT) < 5.0


def test_direct_assignment_bypassing_post_init():
    # Scripts assign config.fitting_model after construction; the fitter must
    # still resolve the preset (background + beta included).
    fitter = make_fitter("lorentzian", "prm0")
    fitter.sample_config.fitting_model = "prm0"
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    # prm0 resolves to the flat baseline, which carries no fit_kind suffix.
    assert result.model == "2lorentzian_x_psf_window"


# ---------------- the model-mixing guard ----------------

def test_mixing_pr_sample_with_lorentzian_reference_raises():
    fitter = make_fitter("prm1", "lorentzian")
    px, sline = make_spectrum()
    with pytest.raises(ValueError, match="[Mm]odel mixing"):
        fitter.fit(px, sline, is_reference_mode=False)


def test_mixing_lorentzian_sample_with_pr_reference_raises():
    fitter = make_fitter("lorentzian", "lorentzian_x_psf")
    px, sline = make_spectrum()
    with pytest.raises(ValueError, match="[Mm]odel mixing"):
        fitter.fit(px, sline, is_reference_mode=False)


def test_camera_constants_are_global():
    # The old per-section camera constants (and their mismatch guard) are
    # gone: one camera, one kernel, shared by sample and reference fits.
    fitter = make_fitter("prm1", "prm1")
    assert fitter.sline_config.psf_tau_left_px == TAU_LEFT
    assert not hasattr(fitter.sample_config, "psf_sigma_px")
    assert not hasattr(fitter.reference_config, "psf_sigma_px")


def test_removed_shared_sigma_raises_with_rename_hint():
    # No aliases (repo rule): the shared psf_sigma_px was SPLIT per peak
    # 2026-08-31 — old code must fail loudly with the new names.
    fitter = make_fitter("prm1", "prm1")
    with pytest.raises(AttributeError, match="psf_sigma_left_px"):
        fitter.sline_config.psf_sigma_px


def test_legacy_shared_sigma_toml_maps_to_both_sides(tmp_path):
    # Stored-data compatibility: a pre-2026-08-31 TOML carries ONE shared
    # psf_sigma_px — it loads onto both per-peak fields.
    from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
        load_sline_from_frame_config)
    toml = tmp_path / "find_peaks_config.toml"
    toml.write_text(
        "[global]\n"
        "pixel_offset_left = 0\n"
        "pixel_offset_right = 0\n"
        "selected_rows = [1, 2]\n"
        "psf_sigma_px = 0.31\n",
        encoding="utf-8")
    cfg = load_sline_from_frame_config(toml)
    assert cfg.psf_sigma_left_px == 0.31
    assert cfg.psf_sigma_right_px == 0.31


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


# ---------------- flat_shared: the calibration baseline ----------------

def test_flat_shared_single_offset_for_reference_fits():
    """ONE shared offset over both windows (the calibration baseline,
    2026-08-25): per-peak offsets trade against the sideband WIDTHS at
    the sweep edges, bending the width-vs-px calibration polynomial and
    fabricating ~+10 MHz of AS-S linewidth gap at the sample position.
    flat_shared removes that per-side freedom: 6 peak parameters plus
    exactly one background value."""
    fitter = make_fitter("lorentzian_x_psf", "lorentzian_x_psf")
    fitter.update_reference_config(replace(
        fitter.reference_config, background="flat_shared"))
    px = np.arange(0, 86, dtype=float)
    true = (psf_profile(px, AMP, CEN_LEFT, GAMMA, SIGMA, TAU_LEFT)
            + psf_profile(px, AMP, CEN_RIGHT, GAMMA, SIGMA, TAU_RIGHT)
            + 80.0)
    rng = np.random.default_rng(3)
    sline = true + rng.normal(0.0, 2.0, size=true.shape)
    result = fitter.fit(px, sline, is_reference_mode=True)
    assert result.is_success
    assert "flat_shared" in result.model
    assert len(result.parameters) == 7      # 2 x 3 peak + ONE offset
    assert abs(float(result.parameters[6]) - 80.0) < 5.0
    assert abs(result.left_peak_width_px - GAMMA) < 0.05
    assert abs(result.right_peak_width_px - GAMMA) < 0.05


def test_packaged_reference_background_is_flat():
    # The shipped [reference] config carries per-sideband offsets ('flat')
    # — the production convention (user decision 2026-08-27, superseding
    # the 08-25 flat_shared choice; on current-era data flat gives L-R
    # width gaps near zero). NOTE: absolute widths are convention-pinned —
    # comparisons against the Figure 4(d) workbook chain (flat_shared era)
    # need a same-scan refit anchor.
    from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
        FIND_PEAKS_TOML_PATH, load_config_section)
    cfg = load_config_section(FIND_PEAKS_TOML_PATH, "reference")
    assert cfg.background == "flat"
