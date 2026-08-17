from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import curve_fit

from brillouin_system.spectrum_fitting.na_correction5 import pupil_angle_limit
from brillouin_system.spectrum_fitting.na_lineshape import (
    make_na_lorentzian,
    na_angular_grid,
    na_mean_shift_ratio,
)


# 20X objective geometry (na_correction5 __main__), clear aqueous sample.
# Uniform-pupil (paper) model: only alpha, no Gaussian coupling.
ALPHA = pupil_angle_limit(8.4, 10.0, 1.328)
R = 0.0            # elastic-line pixel
CENTER_180 = 21.0  # true 180-degree peak pixel (~5 GHz at 0.24 GHz/px)
GAMMA = 1.2
AMP = 1000.0
OFFSET = 50.0
DISP = 0.24        # GHz/px, for MHz reporting only


def _plain_lorentzian(x, a, c, g, o):
    return a * g**2 / ((x - c) ** 2 + g**2) + o


def make_data(seed=0):
    model = make_na_lorentzian(R, ALPHA, n_quad=61)
    px = np.arange(10, 33, dtype=float)
    true = model(px, AMP, CENTER_180, GAMMA, OFFSET)
    rng = np.random.default_rng(seed)
    data = true + rng.normal(0.0, np.sqrt(np.clip(true, 1.0, None)) * 0.3)
    return px, data, model


def test_na_model_recovers_180_degree_position():
    px, data, model = make_data()
    popt, _ = curve_fit(
        model, px, data, p0=[AMP, 20.5, 1.0, OFFSET],
        bounds=([0, 15, 0.1, -np.inf], [np.inf, 25, 5, np.inf]), maxfev=20000,
    )
    na_center = popt[1]
    assert abs(na_center - CENTER_180) * DISP < 0.006  # < 6 MHz


def test_plain_lorentzian_is_biased_low():
    """A symmetric fit to the asymmetric NA peak lands below f180 by ~the NA
    downshift — this is the bias the NA model removes."""
    px, data, _ = make_data()
    p2, _ = curve_fit(_plain_lorentzian, px, data, p0=[AMP, 20.5, 1.2, OFFSET], maxfev=20000)
    lor_center = p2[1]

    v, w, frac = na_angular_grid(ALPHA, 61)
    predicted_downshift_px = (
        np.trapezoid(w * frac, v) / np.trapezoid(w, v)
    ) * (CENTER_180 - R)

    assert lor_center < CENTER_180  # biased low
    # The bias tracks the predicted NA downshift (within noise / shape effects)
    assert abs((CENTER_180 - lor_center) - predicted_downshift_px) * DISP < 0.01


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


def test_zero_na_recovers_plain_lorentzian():
    """In the alpha -> 0 limit the kernel collapses to a single Lorentzian."""
    model = make_na_lorentzian(R, alpha=1e-6, n_quad=11)
    px = np.arange(10, 33, dtype=float)
    na = model(px, AMP, CENTER_180, GAMMA, OFFSET)
    plain = _plain_lorentzian(px, AMP, CENTER_180, GAMMA, OFFSET)
    np.testing.assert_allclose(na, plain, rtol=1e-3, atol=1e-6)
