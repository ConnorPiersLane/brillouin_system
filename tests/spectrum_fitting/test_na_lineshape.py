import numpy as np
import pytest
from scipy.optimize import curve_fit

from brillouin_system.spectrum_fitting.na_correction5 import pupil_angle_limit
from brillouin_system.spectrum_fitting.na_lineshape import (
    make_na_lorentzian,
    na_angular_grid,
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


def test_zero_na_recovers_plain_lorentzian():
    """In the alpha -> 0 limit the kernel collapses to a single Lorentzian."""
    model = make_na_lorentzian(R, alpha=1e-6, n_quad=11)
    px = np.arange(10, 33, dtype=float)
    na = model(px, AMP, CENTER_180, GAMMA, OFFSET)
    plain = _plain_lorentzian(px, AMP, CENTER_180, GAMMA, OFFSET)
    np.testing.assert_allclose(na, plain, rtol=1e-3, atol=1e-6)
