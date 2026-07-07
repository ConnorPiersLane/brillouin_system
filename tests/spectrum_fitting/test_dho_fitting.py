import numpy as np
import pytest

from brillouin_system.spectrum_fitting.dho_model import (
    _2dho_binned,
    dho_intensity,
    dho_peak_offset,
    dho_peak_height,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FindPeaksConfig
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


# Ground truth for the synthetic spectrum. Rayleigh anchors sit outside the
# fitted pixel window so only the inner Stokes / anti-Stokes pair is visible,
# as on the real spectrometer.
RAYLEIGH_LEFT = 120.0
RAYLEIGH_RIGHT = 392.0
OMEGA0 = 60.0
GAMMA = 8.0
AMP_LEFT = 8000.0
AMP_RIGHT = 6000.0
OFFSET = 50.0

U_PK = dho_peak_offset(OMEGA0, GAMMA)
TRUE_CEN_LEFT = RAYLEIGH_LEFT + U_PK
TRUE_CEN_RIGHT = RAYLEIGH_RIGHT - U_PK


def make_config(model: str) -> FindPeaksConfig:
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=100,
        rel_height=0.5,
        wlen_pixels=40,
        fitting_model=model,
        beta=4.0,
    )


def make_synthetic_dho_sline(seed=0):
    px = np.arange(150, 380, dtype=float)
    signal = (
        AMP_LEFT * dho_intensity(px - RAYLEIGH_LEFT, OMEGA0, GAMMA)
        + AMP_RIGHT * dho_intensity(px - RAYLEIGH_RIGHT, OMEGA0, GAMMA)
        + OFFSET
    )
    rng = np.random.default_rng(seed)
    noisy = signal + rng.normal(0.0, np.sqrt(np.clip(signal, 1.0, None)))
    return px, noisy


def fit_with_model(model: str, px, sline):
    fitter = SpectrumFitter()
    fitter.update_sample_config(make_config(model))
    return fitter.fit(px, sline, is_reference_mode=False)


@pytest.mark.parametrize("model", ["dho", "dho_window"])
def test_dho_fit_recovers_peak_positions_and_width(model):
    px, sline = make_synthetic_dho_sline()

    result = fit_with_model(model, px, sline)

    assert result.is_success
    assert result.model == ("2dho" if model == "dho" else "2dho_window")

    assert result.left_peak_center_px == pytest.approx(TRUE_CEN_LEFT, abs=0.5)
    assert result.right_peak_center_px == pytest.approx(TRUE_CEN_RIGHT, abs=0.5)
    assert result.inter_peak_distance == pytest.approx(
        TRUE_CEN_RIGHT - TRUE_CEN_LEFT, abs=0.5
    )

    # Reported width is the HWHM ~ gamma / 2
    assert result.left_peak_width_px == pytest.approx(GAMMA / 2, rel=0.2)
    assert result.right_peak_width_px == result.left_peak_width_px

    # Reported amplitudes are peak heights
    true_height_left = AMP_LEFT * dho_peak_height(OMEGA0, GAMMA)
    true_height_right = AMP_RIGHT * dho_peak_height(OMEGA0, GAMMA)
    assert result.left_peak_amplitude == pytest.approx(true_height_left, rel=0.1)
    assert result.right_peak_amplitude == pytest.approx(true_height_right, rel=0.1)

    assert result.offset == pytest.approx(OFFSET, abs=10.0)


def test_dho_fit_fails_gracefully_with_one_peak():
    px = np.arange(0, 200, dtype=float)
    sline = 1000.0 * np.exp(-0.5 * ((px - 100.0) / 3.0) ** 2) + 20.0

    result = fit_with_model("dho", px, sline)

    assert not result.is_success


def test_2dho_binned_model_is_consistent_with_kernel():
    px = np.arange(150, 380, dtype=float)
    params = dict(amp1=AMP_LEFT, cen1=TRUE_CEN_LEFT, amp2=AMP_RIGHT,
                  cen2=TRUE_CEN_RIGHT, omega0=OMEGA0, gamma=GAMMA, offset=OFFSET)
    binned = _2dho_binned(px, **params)

    unbinned = (
        AMP_LEFT * dho_intensity(px - RAYLEIGH_LEFT, OMEGA0, GAMMA)
        + AMP_RIGHT * dho_intensity(px - RAYLEIGH_RIGHT, OMEGA0, GAMMA)
        + OFFSET
    )

    # Pixel integration smooths the curve slightly; agreement should be close.
    np.testing.assert_allclose(binned, unbinned, rtol=0.05, atol=1.0)


def test_lorentzian_window_regression_for_calibration_pipeline():
    """The calibration pipeline fits with lorentzian_window; make sure the
    DHO additions did not disturb that code path."""
    px = np.arange(0, 512, dtype=float)
    cen1, cen2, wid = 180.0, 330.0, 3.0
    signal = (
        2000.0 / (1.0 + ((px - cen1) / wid) ** 2)
        + 1500.0 / (1.0 + ((px - cen2) / wid) ** 2)
        + 30.0
    )
    rng = np.random.default_rng(1)
    sline = signal + rng.normal(0.0, np.sqrt(np.clip(signal, 1.0, None)))

    result = fit_with_model("lorentzian_window", px, sline)

    assert result.is_success
    assert result.model == "2lorentzian_window"
    assert result.left_peak_center_px == pytest.approx(cen1, abs=0.5)
    assert result.right_peak_center_px == pytest.approx(cen2, abs=0.5)
    assert result.left_peak_width_px == pytest.approx(wid, rel=0.2)
