"""Fit-option resolution and the lineshapes that remain after the
parametric camera PSF was removed (2026-09-10): legacy names normalise,
retired models refuse with a hint, the reference baseline options work on a
plain Lorentzian, and old TOMLs still load."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
from dataclasses import replace

from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FIND_PEAKS_TOML_PATH, FITTING_MODELS_REFERENCE, FITTING_MODELS_SAMPLE,
    FindPeaksConfig, load_config_section, load_sline_from_frame_config)
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SUPPORTED_MODELS, SpectrumFitter, normalize_model_name)

from synthetic_lines import asym_line

SIGMA, GAMMA, AMP = 0.0, 1.0, 3000.0
CEN_LEFT, CEN_RIGHT = 30.0, 56.0


def make_config(model="lorentzian", **kw) -> FindPeaksConfig:
    return FindPeaksConfig(prominence_fraction=0.05, min_peak_width=1, min_peak_height=50,
                           rel_height=0.5, wlen_pixels=20, fitting_model=model, **kw)


def make_fitter(sample="lorentzian", reference="lorentzian") -> SpectrumFitter:
    f = SpectrumFitter()
    f.update_sline_config(replace(f.sline_config, n_peaks=2))
    f.update_sample_config(make_config(sample))
    f.update_reference_config(make_config(reference))
    return f


def make_spectrum(seed=0):
    px = np.arange(0, 86, dtype=float)
    true = (asym_line(px, AMP, CEN_LEFT, GAMMA, SIGMA, 0.0)
            + asym_line(px, AMP, CEN_RIGHT, GAMMA, SIGMA, 0.0) + 80.0)
    rng = np.random.default_rng(seed)
    return px, true + rng.normal(0.0, 2.0, size=true.shape)


def test_the_remaining_models():
    assert SUPPORTED_MODELS == ("lorentzian", "dho_x_psf")
    assert FITTING_MODELS_SAMPLE == ["lorentzian", "dho_x_psf"]
    assert FITTING_MODELS_REFERENCE == ["lorentzian"]


@pytest.mark.parametrize("name", ["lorentzian_x_psf", "prm0", "prm1", "prmr", "pixel_response"])
def test_parametric_psf_models_are_retired_with_a_hint(name):
    with pytest.raises(ValueError, match="removed"):
        make_config(name)


def test_removed_models_raise_with_migration_hint():
    with pytest.raises(ValueError, match="removed"):
        make_config("voigt")
    with pytest.raises(ValueError, match="removed"):
        make_config("na_lorentzian")


def test_legacy_names_normalise():
    cfg = make_config("lorentzian", background="linear_per_peak")
    assert cfg.background == "linear"
    cfg = make_config("lorentzian_window")
    assert cfg.fitting_model == "lorentzian" and cfg.use_window is True
    assert normalize_model_name("lorentzian_window") == ("lorentzian", True)
    assert normalize_model_name("dho_x_psf") == ("dho_x_psf", False)


def test_direct_assignment_bypassing_post_init():
    # scripts assign config.fitting_model after construction; the fitter
    # still resolves the legacy window suffix
    fitter = make_fitter()
    fitter.sample_config.fitting_model = "lorentzian_window"
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    assert result.model == "2lorentzian_window"
    assert abs(result.left_peak_center_px - CEN_LEFT) < 0.03
    assert abs(result.left_peak_width_px - GAMMA) < 0.05


def test_flat_shared_single_offset_for_reference_fits():
    """ONE shared offset over both windows (the calibration baseline
    option, 2026-08-25): 6 peak parameters plus exactly one background
    value."""
    fitter = make_fitter()
    fitter.update_reference_config(replace(fitter.reference_config, background="flat_shared"))
    px, sline = make_spectrum(seed=3)
    result = fitter.fit(px, sline, is_reference_mode=True)
    assert result.is_success
    assert "flat_shared" in result.model
    assert len(result.parameters) == 7      # 2 x 3 peak + ONE offset
    assert abs(float(result.parameters[6]) - 80.0) < 5.0
    assert abs(result.left_peak_width_px - GAMMA) < 0.05
    assert abs(result.right_peak_width_px - GAMMA) < 0.05


def test_packaged_reference_config():
    cfg = load_config_section(FIND_PEAKS_TOML_PATH, "reference")
    assert cfg.fitting_model == "lorentzian"
    assert cfg.background == "flat"


def test_old_tomls_with_the_retired_keys_still_load(tmp_path):
    toml = tmp_path / "old.toml"
    toml.write_text(
        "[global]\npixel_offset_left = 0\npixel_offset_right = 0\n"
        "selected_rows = [1, 2]\npsf_sigma_px = 0.31\npsf_tau_left_px = 0.4\n"
        "psf_box_outer_left_px = 1.95\npsf_sat_ratio_outer_right = 0.037\n"
        "kernel_check = true\n"
        "[sample]\nprominence_fraction = 0.05\nmin_peak_width = 1\nmin_peak_height = 5\n"
        "rel_height = 0.5\nwlen_pixels = 10\nfitting_model = \"dho_x_psf\"\n"
        "dho_kernel = \"measured\"\n"
        "[reference]\nprominence_fraction = 0.05\nmin_peak_width = 1\nmin_peak_height = 800\n"
        "rel_height = 0.5\nwlen_pixels = 40\nfitting_model = \"lorentzian\"\n"
        "centre_method = \"template\"\n", encoding="utf-8")
    g = load_sline_from_frame_config(toml)
    assert g.selected_rows == [1, 2] and not hasattr(g, "psf_sigma_left_px")
    assert load_config_section(toml, "sample").fitting_model == "dho_x_psf"
    assert load_config_section(toml, "reference").fitting_model == "lorentzian"
