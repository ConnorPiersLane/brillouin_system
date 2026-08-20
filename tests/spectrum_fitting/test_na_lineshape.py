"""Tests for the post-hoc NA correction (na_mean_shift_ratio + the angular
grid). The in-fit NA lineshape models were removed 2026-08-20 — the scalar
correction is the production route.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from brillouin_system.spectrum_fitting.na_lineshape import (
    na_angular_grid,
    na_mean_shift_ratio,
    pupil_angle_limit,
)


# 20X objective geometry (8.4 mm pupil, f = 10 mm), clear aqueous sample.
ALPHA = pupil_angle_limit(8.4, 10.0, 1.328)


def test_default_weight_is_uniform_no_gaussian():
    """Default angular weight is the paper's uniform pupil (solid angle sin(v))
    with no Gaussian coupling factor."""
    v, w, _ = na_angular_grid(ALPHA, 41)
    np.testing.assert_allclose(w, np.sin(v))
    # Passing v0 opts INTO the Gaussian apodization (not the default)
    _, w_gauss, _ = na_angular_grid(ALPHA, 41, v0=np.radians(15.0))
    assert np.any(w_gauss < w)  # apodization suppresses large angles


def _ratio_config(weighting, na=0.42, beam_d=6.0, focal=10.0, n=1.33):
    return SimpleNamespace(
        na_weighting=weighting,
        na_collection=na,
        na_beam_diameter_mm=beam_d,
        na_focal_length_mm=focal,
        na_n_sample=n,
    )


def test_mean_shift_ratio_none_is_exactly_one():
    # "none" = no NA correction, an explicit config choice: exactly 1.0 and
    # the other na_* fields are ignored (all-zero here would otherwise raise).
    ratio = na_mean_shift_ratio(
        _ratio_config("none", na=0.0, beam_d=0.0, focal=0.0))
    assert ratio == 1.0


def test_mean_shift_ratio_uniform_na014_is_3p5_mhz_on_water():
    """The paper's low-NA number: uniform hard pupil at NA 0.14 corrects a
    5.07 GHz water shift by about +3.5 MHz (ratio ~ 1 - alpha^2/16)."""
    ratio = na_mean_shift_ratio(_ratio_config("uniform", na=0.14))
    assert 0.0 < ratio < 1.0
    delta_mhz = (5.07 / ratio - 5.07) * 1e3
    assert delta_mhz == pytest.approx(3.5, abs=0.3)


def test_mean_shift_ratio_gauss_smaller_correction_than_uniform():
    """At NA 0.42 the Gaussian apodization suppresses large angles, so its
    correction is smaller than the (overcorrecting) uniform pupil's."""
    r_uniform = na_mean_shift_ratio(_ratio_config("uniform"))
    r_gauss = na_mean_shift_ratio(_ratio_config("uniform_gaussian"))
    assert r_uniform < r_gauss < 1.0
    delta_mhz = (5.07 / r_gauss - 5.07) * 1e3
    # ~+14 MHz at D = 6.0 mm — the documented +13..+16 MHz 20X regime
    assert 10.0 < delta_mhz < 20.0


def test_mean_shift_ratio_validates_config():
    with pytest.raises(ValueError, match="na_collection"):
        na_mean_shift_ratio(_ratio_config("uniform", na=0.0))
    with pytest.raises(ValueError, match="na_beam_diameter_mm"):
        na_mean_shift_ratio(_ratio_config("uniform_gaussian", beam_d=0.0))
    with pytest.raises(ValueError, match="na_weighting"):
        na_mean_shift_ratio(_ratio_config("gaussian"))
