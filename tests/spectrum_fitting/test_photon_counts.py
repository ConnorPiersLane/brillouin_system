import numpy as np
import pytest

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import (
    SENSITIVITY_E_PER_COUNT_PREAMP_1X,
    calculate_photon_counts_from_fitted_spectrum,
    count_to_electrons,
    electrons_per_count,
)


def test_sensitivity_matches_photon_transfer_measurement():
    """Measured 2026-07-27 on the DU897: 3.5 +- 0.5 e-/count at preamp 1x."""
    assert 3.0 <= SENSITIVITY_E_PER_COUNT_PREAMP_1X <= 4.0


def test_conventional_mode_uses_measured_sensitivity():
    # Conventional mode reports emccd_gain == 0, preamp multiplier 1x.
    assert electrons_per_count(preamp_gain=1.0, emccd_gain=0) == pytest.approx(
        SENSITIVITY_E_PER_COUNT_PREAMP_1X
    )


def test_higher_preamp_multiplier_lowers_electrons_per_count():
    """A larger preamp multiplier digitises the same charge into more counts,
    so each count represents FEWER electrons."""
    one_x = electrons_per_count(preamp_gain=1.0, emccd_gain=0)
    two_x = electrons_per_count(preamp_gain=2.0, emccd_gain=0)
    assert two_x == pytest.approx(one_x / 2.0)
    assert two_x < one_x


def test_em_gain_divides_electrons_per_count():
    no_em = electrons_per_count(preamp_gain=1.0, emccd_gain=0)
    with_em = electrons_per_count(preamp_gain=1.0, emccd_gain=100)
    assert with_em == pytest.approx(no_em / 100.0)


def test_count_to_electrons_scales_linearly():
    assert count_to_electrons(1000, preamp_gain=1.0, emccd_gain=0) == pytest.approx(
        1000 * SENSITIVITY_E_PER_COUNT_PREAMP_1X
    )


def _fs(amp_l=100.0, wid_l=1.2, amp_r=110.0, wid_r=1.0):
    return FittedSpectrum(
        is_success=True,
        x_pixels=np.arange(10),
        sline=np.zeros(10),
        left_peak_amplitude=amp_l,
        left_peak_width_px=wid_l,
        right_peak_amplitude=amp_r,
        right_peak_width_px=wid_r,
    )


def test_photon_counts_use_the_area_of_each_peak():
    fs = _fs()
    p = calculate_photon_counts_from_fitted_spectrum(fs, preamp_gain=1.0, emccd_gain=0)
    expected_left = np.pi * 100.0 * 1.2 * SENSITIVITY_E_PER_COUNT_PREAMP_1X
    assert p.left_peak_photons == pytest.approx(expected_left)
    assert p.total_photons == pytest.approx(p.left_peak_photons + p.right_peak_photons)


def test_failed_fit_returns_no_photons():
    fs = FittedSpectrum(is_success=False, x_pixels=np.arange(3), sline=np.zeros(3))
    p = calculate_photon_counts_from_fitted_spectrum(fs, preamp_gain=1.0, emccd_gain=0)
    assert p.left_peak_photons is None
    assert p.total_photons is None


def test_sensitivity_override_is_respected():
    p = calculate_photon_counts_from_fitted_spectrum(
        _fs(), preamp_gain=1.0, emccd_gain=0, sensitivity_e_per_count=1.0
    )
    assert p.left_peak_photons == pytest.approx(np.pi * 100.0 * 1.2)


def test_lorentzian_crlb_factor_is_two():
    """Thompson's photon term assumes a Gaussian (bound s/sqrt(N)); a Lorentzian
    of HWHM g has bound g*sqrt(2/N), so the variance carries a factor 2."""
    from brillouin_system.spectrum_fitting.spectrum_analyzer import (
        LORENTZIAN_CRLB_FACTOR,
    )

    assert LORENTZIAN_CRLB_FACTOR == pytest.approx(2.0)
