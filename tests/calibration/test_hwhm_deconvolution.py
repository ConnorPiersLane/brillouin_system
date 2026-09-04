"""Widths are reported as three separate quantities, never one field whose
meaning depends on the model: the raw fitted HWHM, the instrument HWHM from the
calibration sidebands, and the sample linewidth left after subtracting them.
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

PRM_FIT = "2pixel_response_window_linear_per_peak"


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


@pytest.mark.parametrize("model", [PRM_FIT, "2lorentzian_window"])
def test_fitted_hwhm_is_raw_whatever_the_model(model):
    """The meaning of hwhm_ghz never depends on the lineshape."""
    left, right = make_calc().hwhm_ghz(make_fit(model))

    expected = SLOPE * FIT_WIDTH_PX
    assert left == pytest.approx(expected)
    assert right == pytest.approx(expected)


def test_instrument_hwhm_is_read_at_the_sample_peak_pixel():
    left, right = make_calc().instrument_hwhm_ghz(20.0, 60.0)

    expected = SLOPE * INSTRUMENT_PX
    assert left == pytest.approx(expected)
    assert right == pytest.approx(expected)


def test_sample_linewidth_subtracts_the_instrument_width():
    left, right = make_calc().sample_linewidth_ghz(make_fit(PRM_FIT))

    expected = SLOPE * (FIT_WIDTH_PX - INSTRUMENT_PX)
    assert left == pytest.approx(expected)
    assert right == pytest.approx(expected)


def test_no_sample_linewidth_for_other_lineshapes():
    """Only the pixel-response fit is the validated width recipe."""
    left, right = make_calc().sample_linewidth_ghz(make_fit("2lorentzian_window"))

    assert left is None and right is None


def test_no_sample_linewidth_without_a_calibration_width_model():
    """Old scans have no width polynomial, so there is no instrument term."""
    calc = make_calc(with_width_model=False)

    assert calc.instrument_hwhm_ghz(20.0, 60.0) == (None, None)
    assert calc.sample_linewidth_ghz(make_fit(PRM_FIT)) == (None, None)


def test_failed_fit_reports_none():
    fit = FittedSpectrum(is_success=False, x_pixels=np.arange(80.0), sline=np.zeros(80))
    calc = make_calc()

    assert calc.hwhm_ghz(fit) == (None, None)
    assert calc.sample_linewidth_ghz(fit) == (None, None)
