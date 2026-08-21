import math

import numpy as np
import pytest

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationPolyfitParameters,
)
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.noise_analysis import (
    LORENTZIAN_PHOTON_FACTOR,
    PixelCountsAndPhotons,
    SENSITIVITY_E_PER_COUNT_PREAMP_1X,
    count_to_electrons,
    electrons_per_count,
    theoretical_precision,
)


def test_sensitivity_matches_photon_transfer_measurement():
    """Measured 2026-08-12 on the DU897: 3.89 +- 0.04 e-/count at preamp 1x
    (quadratic photon transfer; a linear fit is biased low by common-mode
    source noise and gave the earlier 3.5)."""
    assert SENSITIVITY_E_PER_COUNT_PREAMP_1X == pytest.approx(3.89)


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
    """EM sensitivity must be supplied explicitly (see the EM guard tests)."""
    no_em = electrons_per_count(preamp_gain=1.0, emccd_gain=0,
                                sensitivity_e_per_count=5.0)
    with_em = electrons_per_count(preamp_gain=1.0, emccd_gain=100,
                                  sensitivity_e_per_count=5.0)
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


def test_counts_and_photons_use_the_area_of_each_peak():
    p = PixelCountsAndPhotons.from_fit(_fs(), preamp_gain=1.0, emccd_gain=0)
    # Counts: exact pixel-integrated Lorentzian area pi * amp * width.
    assert p.left_peak_counts == pytest.approx(np.pi * 100.0 * 1.2)
    assert p.right_peak_counts == pytest.approx(np.pi * 110.0 * 1.0)
    # Photons: counts scaled by the measured sensitivity.
    assert p.left_peak_photons == pytest.approx(
        p.left_peak_counts * SENSITIVITY_E_PER_COUNT_PREAMP_1X)
    assert p.total_counts == pytest.approx(p.left_peak_counts + p.right_peak_counts)
    assert p.total_photons == pytest.approx(p.left_peak_photons + p.right_peak_photons)


def test_failed_fit_returns_no_counts_or_photons():
    fs = FittedSpectrum(is_success=False, x_pixels=np.arange(3), sline=np.zeros(3))
    p = PixelCountsAndPhotons.from_fit(fs, preamp_gain=1.0, emccd_gain=0)
    assert p.left_peak_counts is None
    assert p.left_peak_photons is None
    assert p.total_photons is None


def test_sensitivity_override_is_respected():
    p = PixelCountsAndPhotons.from_fit(
        _fs(), preamp_gain=1.0, emccd_gain=0, sensitivity_e_per_count=1.0
    )
    assert p.left_peak_photons == pytest.approx(np.pi * 100.0 * 1.2)


def test_lorentzian_crlb_factor_is_two():
    """Thompson's photon term assumes a Gaussian (bound s/sqrt(N)); a Lorentzian
    of HWHM g has bound g*sqrt(2/N), so the variance carries a factor 2."""
    assert LORENTZIAN_PHOTON_FACTOR == pytest.approx(2.0)


def _linear_calculator():
    """Left order rises with px, right order falls, distance in between —
    matching the real instrument's opposite-sign dispersions."""
    return CalibrationCalculator(CalibrationPolyfitParameters(
        degree=1,
        freq_left_peak=np.array([0.28, 0.0]),
        freq_right_peak=np.array([-0.35, 30.0]),
        freq_peak_distance=np.array([-0.156, 10.0]),
    ))


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
        # The fit carries its own row band (single-source rule): the bound
        # reads n_rows from here for read noise and the dark-level offset.
        sline_rows=list(range(13)),
    )


def _bound(fs, photons, calc, corr=0.0):
    return theoretical_precision(
        fs=fs, photons=photons, calibration_calculator=calc,
        preamp_gain=1.0, emccd_gain=0, corr_left_right=corr)


def test_distance_precision_combines_the_two_orders_in_quadrature():
    """With uncorrelated peaks, var(c_R - c_L) = var(c_R) + var(c_L)."""
    calc = _linear_calculator()
    fs = _fitted_two_peaks()
    photons = PixelCountsAndPhotons.from_fit(fs, preamp_gain=1.0, emccd_gain=0)
    t = _bound(fs, photons, calc)

    a_l, a_r, a_d = 0.28, 0.35, 0.156
    expected = 1e3 * a_d * math.hypot(
        t.left_peak_total_mhz / 1e3 / a_l, t.right_peak_total_mhz / 1e3 / a_r
    )
    assert t.distance_total_mhz == pytest.approx(expected, rel=1e-6)


def test_distance_is_tighter_than_either_single_order():
    calc = _linear_calculator()
    fs = _fitted_two_peaks()
    photons = PixelCountsAndPhotons.from_fit(fs, preamp_gain=1.0, emccd_gain=0)
    t = _bound(fs, photons, calc)

    assert t.distance_total_mhz < t.left_peak_total_mhz
    assert t.distance_total_mhz < t.right_peak_total_mhz


def test_anticorrelation_widens_the_distance_uncertainty():
    """cov enters with a minus sign, so negative correlation inflates it."""
    calc = _linear_calculator()
    fs = _fitted_two_peaks()
    photons = PixelCountsAndPhotons.from_fit(fs, preamp_gain=1.0, emccd_gain=0)

    indep = _bound(fs, photons, calc).distance_total_mhz
    anti = _bound(fs, photons, calc, corr=-0.2).distance_total_mhz
    pos = _bound(fs, photons, calc, corr=0.2).distance_total_mhz
    assert anti > indep > pos


def test_pedestal_shot_noise_widens_the_bound():
    """The fitted background level under a peak feeds the b term (Poisson),
    so a brighter stray pedestal must widen the bound — from the fit alone.
    Raw-frame fits: the fitted background contains the camera dark level
    (ccd dark_median_counts * n_rows), which the bound subtracts before
    the Poisson term, so the pedestal here sits ON TOP of that level."""
    from brillouin_system.ccd_characteristics import ccd_config

    calc = _linear_calculator()
    fs_clean = _fitted_two_peaks()
    from dataclasses import replace
    dark_level = ccd_config.get().dark_median_counts * len(fs_clean.sline_rows)
    fs_pedestal = replace(fs_clean,
                          left_peak_bg_counts=dark_level + 500.0,
                          right_peak_bg_counts=dark_level + 500.0)
    photons = PixelCountsAndPhotons.from_fit(fs_clean, preamp_gain=1.0,
                                             emccd_gain=0)
    clean = _bound(fs_clean, photons, calc)
    pedestal = _bound(fs_pedestal, photons, calc)
    assert pedestal.left_peak_bg_mhz > clean.left_peak_bg_mhz
    assert pedestal.distance_total_mhz > clean.distance_total_mhz


def test_dark_level_carries_no_shot_noise():
    """A fitted background at exactly the camera dark level is an
    electronic offset, not light: the bound must treat it like no pedestal
    at all (read noise only)."""
    from brillouin_system.ccd_characteristics import ccd_config
    from dataclasses import replace

    calc = _linear_calculator()
    fs_clean = _fitted_two_peaks()
    dark_level = ccd_config.get().dark_median_counts * len(fs_clean.sline_rows)
    fs_dark = replace(fs_clean, left_peak_bg_counts=dark_level,
                      right_peak_bg_counts=dark_level)
    photons = PixelCountsAndPhotons.from_fit(fs_clean, preamp_gain=1.0,
                                             emccd_gain=0)
    clean = _bound(fs_clean, photons, calc)
    dark = _bound(fs_dark, photons, calc)
    assert dark.left_peak_bg_mhz == pytest.approx(clean.left_peak_bg_mhz)


def test_fit_row_band_scales_the_read_noise():
    """The bound reads the row count from the fit itself: four times the
    rows means twice the per-sline-pixel read noise (rn*sqrt(n))."""
    from dataclasses import replace

    calc = _linear_calculator()
    fs4 = replace(_fitted_two_peaks(), sline_rows=list(range(4)))
    fs16 = replace(_fitted_two_peaks(), sline_rows=list(range(16)))
    photons = PixelCountsAndPhotons.from_fit(fs4, preamp_gain=1.0,
                                             emccd_gain=0)
    t4 = _bound(fs4, photons, calc)
    t16 = _bound(fs16, photons, calc)
    assert t16.left_peak_bg_mhz == pytest.approx(2.0 * t4.left_peak_bg_mhz)


def test_detected_width_reduces_to_gamma_without_psf():
    from brillouin_system.spectrum_fitting.psf import detected_hwhm_px
    assert detected_hwhm_px(1.5, 0.0, 0.0) == pytest.approx(1.5)


def test_detected_width_grows_with_the_psf():
    from brillouin_system.spectrum_fitting.psf import detected_hwhm_px
    plain = detected_hwhm_px(1.2, 0.0, 0.0)
    blurred = detected_hwhm_px(1.2, 0.25, 0.4)
    # Production kernel widens a 1.2 px core by a few-to-ten percent.
    assert 1.02 * plain < blurred < 1.25 * plain


def test_psf_fit_bound_uses_the_detected_width():
    """Same fitted widths: a PSF-tagged fit must report a wider (more
    honest) bound than a plain-Lorentzian one, because its photons are
    spread by the camera PSF on top of the fitted core."""
    from dataclasses import replace
    calc = _linear_calculator()
    fs_plain = _fitted_two_peaks()                       # model='' -> plain
    fs_psf = replace(fs_plain, model="2lorentzian_x_psf_window")
    photons = PixelCountsAndPhotons.from_fit(fs_plain, preamp_gain=1.0,
                                             emccd_gain=0)
    t_plain = _bound(fs_plain, photons, calc)
    t_psf = _bound(fs_psf, photons, calc)
    assert t_psf.left_peak_photons_mhz > t_plain.left_peak_photons_mhz
    assert t_psf.distance_total_mhz > t_plain.distance_total_mhz


def test_failed_fit_has_no_distance_precision():
    calc = _linear_calculator()
    fs = FittedSpectrum(is_success=False, x_pixels=np.arange(3), sline=np.zeros(3))
    photons = PixelCountsAndPhotons.from_fit(fs, preamp_gain=1.0, emccd_gain=0)
    t = _bound(fs, photons, calc)
    assert t.distance_total_mhz is None


def test_em_mode_raises_because_its_sensitivity_is_not_calibrated():
    """EM mode reads out through a different amplifier, so the Conventional
    sensitivity does not apply. Better to fail loudly than return a wrong N."""
    with pytest.raises(ValueError, match="Electron-Multiplying"):
        electrons_per_count(preamp_gain=1.0, emccd_gain=100)


def test_em_guard_is_not_bypassed_by_the_public_helpers():
    with pytest.raises(ValueError, match="Electron-Multiplying"):
        count_to_electrons(1000, preamp_gain=1.0, emccd_gain=100)
    with pytest.raises(ValueError, match="Electron-Multiplying"):
        PixelCountsAndPhotons.from_fit(_fs(), preamp_gain=1.0, emccd_gain=100)


def test_em_mode_works_when_a_sensitivity_is_supplied():
    got = electrons_per_count(preamp_gain=1.0, emccd_gain=100,
                              sensitivity_e_per_count=5.0)
    assert got == pytest.approx(5.0 / 100)


def test_invalid_preamp_multiplier_raises():
    with pytest.raises(ValueError, match="preamp multiplier"):
        electrons_per_count(preamp_gain=0, emccd_gain=0)
