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


def _analyzer_with_linear_calibration():
    """Left order rises with px, right order falls, distance in between —
    matching the real instrument's opposite-sign dispersions."""
    import numpy as np

    from brillouin_system.calibration.calibration import (
        CalibrationCalculator,
        CalibrationPolyfitParameters,
    )
    from brillouin_system.spectrum_fitting.spectrum_analyzer import SpectrumAnalyzer

    params = CalibrationPolyfitParameters(
        degree=1,
        freq_left_peak=np.array([0.28, 0.0]),
        freq_right_peak=np.array([-0.35, 30.0]),
        freq_peak_distance=np.array([-0.156, 10.0]),
    )
    return SpectrumAnalyzer(CalibrationCalculator(params))


def _fitted_two_peaks():
    return FittedSpectrum(
        is_success=True,
        x_pixels=np.arange(84),
        sline=np.zeros(84),
        left_peak_center_px=27.0,
        left_peak_width_px=1.2,
        left_peak_amplitude=500.0,
        right_peak_center_px=60.0,
        right_peak_width_px=1.0,
        right_peak_amplitude=520.0,
        inter_peak_distance=33.0,
    )


def test_distance_precision_combines_the_two_orders_in_quadrature():
    """With uncorrelated peaks, var(c_R - c_L) = var(c_R) + var(c_L)."""
    import math

    an = _analyzer_with_linear_calibration()
    fs = _fitted_two_peaks()
    photons = calculate_photon_counts_from_fitted_spectrum(fs, preamp_gain=1.0, emccd_gain=0)
    t = an.theoretical_precision(fs, photons, None, preamp_gain=1.0, emccd_gain=0)

    a_l, a_r, a_d = 0.28, 0.35, 0.156
    expected = 1e3 * a_d * math.hypot(
        t.left_peak_total_mhz / 1e3 / a_l, t.right_peak_total_mhz / 1e3 / a_r
    )
    assert t.distance_total_mhz == pytest.approx(expected, rel=1e-6)


def test_distance_is_tighter_than_either_single_order():
    an = _analyzer_with_linear_calibration()
    fs = _fitted_two_peaks()
    photons = calculate_photon_counts_from_fitted_spectrum(fs, preamp_gain=1.0, emccd_gain=0)
    t = an.theoretical_precision(fs, photons, None, preamp_gain=1.0, emccd_gain=0)

    assert t.distance_total_mhz < t.left_peak_total_mhz
    assert t.distance_total_mhz < t.right_peak_total_mhz


def test_anticorrelation_widens_the_distance_uncertainty():
    """cov enters with a minus sign, so negative correlation inflates it."""
    an = _analyzer_with_linear_calibration()
    fs = _fitted_two_peaks()
    photons = calculate_photon_counts_from_fitted_spectrum(fs, preamp_gain=1.0, emccd_gain=0)

    indep = an.theoretical_precision(fs, photons, None, 1.0, 0).distance_total_mhz
    anti = an.theoretical_precision(fs, photons, None, 1.0, 0,
                                    corr_left_right=-0.2).distance_total_mhz
    pos = an.theoretical_precision(fs, photons, None, 1.0, 0,
                                   corr_left_right=0.2).distance_total_mhz
    assert anti > indep > pos


def test_failed_fit_has_no_distance_precision():
    an = _analyzer_with_linear_calibration()
    fs = FittedSpectrum(is_success=False, x_pixels=np.arange(3), sline=np.zeros(3))
    photons = calculate_photon_counts_from_fitted_spectrum(fs, preamp_gain=1.0, emccd_gain=0)
    t = an.theoretical_precision(fs, photons, None, preamp_gain=1.0, emccd_gain=0)
    assert t.distance_total_mhz is None
