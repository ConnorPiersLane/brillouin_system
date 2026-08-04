"""The reported HWHM of a pixel-response fit is the SAMPLE linewidth: the
instrument width measured from the calibration sidebands is already subtracted.
Every other lineshape keeps reporting the raw, instrument-broadened width.
"""
import numpy as np
import pytest

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationPolyfitParameters,
)
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum

# 1 px = 0.1 GHz on both peaks, and a flat instrument width of 2 px.
SLOPE = 0.1
INSTRUMENT_PX = 2.0
FIT_WIDTH_PX = 5.0


def make_calc(with_width_model: bool = True) -> CalibrationCalculator:
    width = np.array([0.0, INSTRUMENT_PX]) if with_width_model else None
    return CalibrationCalculator(CalibrationPolyfitParameters(
        degree=1,
        freq_left_peak=np.array([-SLOPE, 0.0]),   # left peak: negative slope
        freq_right_peak=np.array([SLOPE, 0.0]),
        freq_peak_distance=np.array([SLOPE, 0.0]),
        calibration_width_left_peak=width,
        calibration_width_right_peak=width,
    ))


def make_fit(model: str) -> FittedSpectrum:
    return FittedSpectrum(
        is_success=True,
        x_pixels=np.arange(80.0),
        sline=np.zeros(80),
        model=model,
        left_peak_center_px=20.0,
        right_peak_center_px=60.0,
        left_peak_width_px=FIT_WIDTH_PX,
        right_peak_width_px=FIT_WIDTH_PX,
    )


def test_pixel_response_hwhm_has_instrument_width_subtracted():
    left, right = make_calc().hwhm_ghz(make_fit("2pixel_response_window_linear_per_peak"))

    expected = SLOPE * (FIT_WIDTH_PX - INSTRUMENT_PX)
    assert left == pytest.approx(expected)
    assert right == pytest.approx(expected)


def test_lorentzian_hwhm_stays_raw():
    left, right = make_calc().hwhm_ghz(make_fit("2lorentzian_window"))

    expected = SLOPE * FIT_WIDTH_PX
    assert left == pytest.approx(expected)
    assert right == pytest.approx(expected)


def test_reference_mode_does_not_subtract():
    """A fit OF the calibration is the instrument; subtracting gives zero."""
    left, right = make_calc().hwhm_ghz(
        make_fit("2pixel_response_window"), deconvolve=False)

    expected = SLOPE * FIT_WIDTH_PX
    assert left == pytest.approx(expected)
    assert right == pytest.approx(expected)


def test_missing_width_model_reports_none_rather_than_the_raw_width():
    """Old scans have no width polynomial. Returning the raw width would pass
    off an instrument-broadened number as a sample linewidth."""
    left, right = make_calc(with_width_model=False).hwhm_ghz(
        make_fit("2pixel_response_window"))

    assert left is None and right is None


def test_failed_fit_reports_none():
    left, right = make_calc().hwhm_ghz(
        FittedSpectrum(is_success=False, x_pixels=np.arange(80.0), sline=np.zeros(80)))

    assert left is None and right is None
