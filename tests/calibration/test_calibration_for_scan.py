"""A scan's calibration is re-fitted from its own raw frames, so the calibration
and the samples share a peak-centre convention. The stored polynomial was fitted
at acquisition time with an unrecorded model, so it cannot back a pixel-response
re-analysis.
"""
import numpy as np
import pytest

from brillouin_system.calibration.calibration import (
    CalibrationData,
    CalibrationMeasurementPoint,
    CalibrationPolyfitParameters,
    MeasurementsPerFreq,
)
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.human_interface_measurements import (
    AxialScan,
    calibration_for_scan,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
    PsfConstants,
    SlineFromFrameConfig,
)
from brillouin_system.spectrum_fitting.psf import psf_profile
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

SIGMA, TAU_L, TAU_R = 0.25, 0.4, 0.2
N_ROWS = 8
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
    fitter.psf_config = PsfConstants(
        psf_sigma_px=SIGMA, psf_tau_left_px=TAU_L,
        psf_tau_right_px=TAU_R)
    fitter.update_sample_config(make_config(model))
    fitter.update_reference_config(make_config(model))
    fitter.update_sline_config(SlineFromFrameConfig(
        pixel_offset_left=0, pixel_offset_right=0,
        selected_rows=list(range(N_ROWS)), row_selection="manual",
    ))
    return fitter


def make_frame(separation_px: float) -> np.ndarray:
    """Two sidebands, drawn on enough rows for the sline to sum over."""
    px = np.arange(0, 86, dtype=float)
    mid = 43.0
    line = (
        psf_profile(px, 3000.0, mid - separation_px / 2, 1.0, SIGMA, TAU_L)
        + psf_profile(px, 3000.0, mid + separation_px / 2, 1.0, SIGMA, TAU_R)
        + 100.0
    )
    return np.tile(line / N_ROWS, (N_ROWS, 1))


def make_calibration_data() -> CalibrationData:
    """A sweep: the sidebands walk apart as the microwave frequency rises."""
    blocks = []
    for freq, sep in [(4.0, 20.0), (6.0, 30.0), (8.0, 40.0)]:
        point = CalibrationMeasurementPoint(
            frame=make_frame(sep), microwave_freq=freq)
        blocks.append(MeasurementsPerFreq(
            set_freq_ghz=freq, cali_meas_points=[point]))
    return CalibrationData(measured_freqs=blocks)


def make_scan(with_frames: bool) -> AxialScan:
    return AxialScan(
        i=0, id="scan-under-test", measurements=[], system_state=None,
        calibration_params=CalibrationPolyfitParameters(
            degree=1,
            freq_left_peak=STORED, freq_right_peak=STORED,
            freq_peak_distance=STORED,
        ),
        calibration_data=make_calibration_data() if with_frames else None,
    )


def test_stored_frames_are_refitted_not_reused():
    calc = calibration_for_scan(make_scan(with_frames=True), make_fitter("prm1"))

    assert not np.allclose(calc.p.freq_left_peak, STORED)
    # A real fit fills in what the stored stub never had.
    assert calc.p.calibration_width_left_peak is not None
    assert calc.p.left_px_points is not None and len(calc.p.left_px_points) == 3


def test_refit_uses_the_scans_row_band():
    """The band must not move between a calibration and its samples."""
    fitter = make_fitter("prm1")
    calibration_for_scan(make_scan(with_frames=True), fitter)

    assert fitter.get_selected_rows() == list(range(N_ROWS))


def test_pixel_response_without_raw_frames_is_refused():
    with pytest.raises(ValueError, match="no raw calibration frames"):
        calibration_for_scan(make_scan(with_frames=False), make_fitter("prm1"))


def test_lorentzian_without_raw_frames_falls_back_to_the_stored_polynomial():
    calc = calibration_for_scan(make_scan(with_frames=False), make_fitter("lorentzian"))

    assert np.allclose(calc.p.freq_left_peak, STORED)
