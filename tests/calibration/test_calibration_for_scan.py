"""A scan's calibration is re-fitted from its own raw frames (the template
chain), so the calibration and the samples share a peak-centre convention.
calibration_calculator_for_scan takes only the scan's calibration
information (calibration_data, calibration_params) plus the fitter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "spectrum_fitting"))

import numpy as np

from brillouin_system.calibration.calibration import (
    CalibrationData,
    CalibrationMeasurementPoint,
    CalibrationPolyfitParameters,
    MeasurementsPerFreq,
    calibration_calculator_for_scan,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
    SlineFromFrameConfig,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

from synthetic_lines import asym_line

SIGMA, TAU_L, TAU_R = 0.25, 0.4, 0.2
N_ROWS = 8
N_FRAMES = 41
STORED = np.array([0.1, 0.0])


def make_config(model: str) -> FindPeaksConfig:
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=50,
        rel_height=0.5,
        wlen_pixels=20,
        fitting_model=model,
    )


def make_fitter(model: str) -> SpectrumFitter:
    fitter = SpectrumFitter()
    fitter.update_sample_config(make_config(model))
    fitter.update_reference_config(make_config("lorentzian"))
    fitter.update_sline_config(SlineFromFrameConfig(
        pixel_offset_left=0, pixel_offset_right=0,
        selected_rows=list(range(N_ROWS)), row_selection="manual",
        envelope_source="config", kernel_source="scan",
    ))
    return fitter


def make_frame(separation_px: float) -> np.ndarray:
    """Two sidebands, drawn on enough rows for the sline to sum over."""
    px = np.arange(0, 86, dtype=float)
    mid = 43.0
    line = (
        asym_line(px, 3000.0, mid - separation_px / 2, 1.0, SIGMA, TAU_L)
        + asym_line(px, 3000.0, mid + separation_px / 2, 1.0, SIGMA, TAU_R)
        + 100.0
    )
    return np.tile(line / N_ROWS, (N_ROWS, 1))


def make_calibration_data() -> CalibrationData:
    """A sweep: the sidebands walk apart as the microwave frequency rises
    (0.25 px per frame per line, dense enough for the template chain)."""
    blocks = []
    for k in range(N_FRAMES):
        freq = 4.0 + 0.1 * k
        sep = 20.0 + 5.0 * (freq - 4.0)
        point = CalibrationMeasurementPoint(
            frame=make_frame(sep), microwave_freq=freq)
        blocks.append(MeasurementsPerFreq(
            set_freq_ghz=freq, cali_meas_points=[point]))
    return CalibrationData(measured_freqs=blocks)


def make_stored_params() -> CalibrationPolyfitParameters:
    return CalibrationPolyfitParameters(
        degree=1,
        freq_left_peak=STORED, freq_right_peak=STORED,
        freq_peak_distance=STORED,
    )


def test_stored_frames_are_refitted_not_reused():
    calc = calibration_calculator_for_scan(
        make_calibration_data(), make_stored_params(), make_fitter("dho_x_psf"))

    assert not np.allclose(calc.p.freq_left_peak, STORED)
    # the re-fit fills in what the stored stub never had: the width tracks
    # and the measured profile the DHO kernels come from
    assert calc.p.calibration_width_left_peak is not None
    assert calc.p.template_profiles is not None
    assert calc.p.left_px_points is not None and len(calc.p.left_px_points) == N_FRAMES
    # the tracks reproduce the synthetic geometry: 2.5 px/GHz per line
    assert abs(calc.dfreq_dpx_left_peak(30.0) - (-1.0 / 2.5)) < 0.02
    assert abs(calc.dfreq_dpx_right_peak(56.0) - (1.0 / 2.5)) < 0.02


def test_refit_uses_the_scans_row_band():
    """The band must not move between a calibration and its samples."""
    fitter = make_fitter("dho_x_psf")
    calibration_calculator_for_scan(
        make_calibration_data(), make_stored_params(), fitter)

    assert fitter.get_selected_rows() == list(range(N_ROWS))


def test_without_raw_frames_the_stored_polynomial_is_used():
    for model in ("lorentzian", "dho_x_psf"):
        calc = calibration_calculator_for_scan(
            None, make_stored_params(), make_fitter(model))
        assert np.allclose(calc.p.freq_left_peak, STORED)
        assert getattr(calc.p, "template_profiles", None) is None
