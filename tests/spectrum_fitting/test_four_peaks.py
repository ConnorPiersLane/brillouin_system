"""Tests for the opt-in 4-peak fit (SpectrumFitter.fit(..., n_peaks=4)):
selection by amplitude ranking, per-position tails (psf_tau_outer_* for the
outer orders), reported left/right = the inner main pair, the outer_* result
fields, and the outer-order frequency tracks + four-peak combination.
"""
import numpy as np
import pytest

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationData,
    CalibrationMeasurementPoint,
    CalibrationPolyfitParameters,
    MeasurementsPerFreq,
)
from brillouin_system.calibration.outer_order_tracks import (
    build_outer_order_tracks,
    four_peak_shift,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
    PsfConstants,
    SlineFromFrameConfig,
)
from brillouin_system.spectrum_fitting.psf import psf_profile
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

SIGMA, TAU_L, TAU_R = 0.25, 0.4, 0.2
TAU_OL, TAU_OR = 0.5, 0.0
# the real 4-peak ROI geometry: S_outer / AS / S / AS_outer
CENTERS = (39.0, 83.0, 118.0, 146.0)
TAUS = (TAU_OL, TAU_L, TAU_R, TAU_OR)
AMPS = (1800.0, 3000.0, 3000.0, 1800.0)   # outer ~60% of the main pair
GAMMA = 1.0
OFFSET = 80.0


def make_config(model="prm0") -> FindPeaksConfig:
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=50,
        rel_height=0.5,
        wlen_pixels=20,
        fitting_model=model,
    )


def make_fitter(model="prm0") -> SpectrumFitter:
    fitter = SpectrumFitter()
    fitter.psf_config = PsfConstants(
        psf_sigma_px=SIGMA, psf_tau_left_px=TAU_L, psf_tau_right_px=TAU_R,
        psf_tau_outer_left_px=TAU_OL, psf_tau_outer_right_px=TAU_OR)
    fitter.update_sample_config(make_config(model))
    fitter.update_reference_config(make_config("lorentzian_x_psf"))
    return fitter


def make_spectrum(seed=0, centers=CENTERS):
    px = np.arange(0.0, 200.0)
    true = np.full_like(px, OFFSET)
    for a, c, tau in zip(AMPS, centers, TAUS):
        true = true + psf_profile(px, a, c, GAMMA, SIGMA, tau)
    rng = np.random.default_rng(seed)
    return px, true + rng.normal(0.0, 2.0, size=true.shape)


def test_n_peaks_validation():
    fitter = make_fitter()
    px, sline = make_spectrum()
    with pytest.raises(ValueError, match="n_peaks"):
        fitter.fit(px, sline, is_reference_mode=False, n_peaks=3)


def test_two_peak_default_picks_the_inner_pair():
    # amplitude ranking: the brightest two are the inner main pair
    fitter = make_fitter()
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False)
    assert result.is_success
    assert result.model.startswith("2")
    assert abs(result.left_peak_center_px - CENTERS[1]) < 0.05
    assert abs(result.right_peak_center_px - CENTERS[2]) < 0.05
    # production two-peak fits carry no outer fields
    assert result.outer_left_peak_center_px is None
    assert result.outer_right_peak_center_px is None


def test_four_peak_fit_recovers_all_orders():
    fitter = make_fitter()
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False, n_peaks=4)
    assert result.is_success
    assert result.model.startswith("4")
    # the REPORTED left/right stay the inner main pair
    assert abs(result.left_peak_center_px - CENTERS[1]) < 0.05
    assert abs(result.right_peak_center_px - CENTERS[2]) < 0.05
    assert abs(result.inter_peak_distance - (CENTERS[2] - CENTERS[1])) < 0.1
    # the outer orders land in the outer_* fields
    assert abs(result.outer_left_peak_center_px - CENTERS[0]) < 0.05
    assert abs(result.outer_right_peak_center_px - CENTERS[3]) < 0.05
    for w in (result.left_peak_width_px, result.right_peak_width_px,
              result.outer_left_peak_width_px,
              result.outer_right_peak_width_px):
        assert abs(w - GAMMA) < 0.05
    # per-peak backgrounds map with their peaks (all background offsets equal here)
    for bg in (result.left_peak_bg_counts, result.right_peak_bg_counts,
               result.outer_left_peak_bg_counts,
               result.outer_right_peak_bg_counts):
        assert abs(bg - OFFSET) < 5.0


def test_four_peak_matches_two_peak_on_main_pair():
    # the main-pair centres must not move between the 2- and 4-peak fits
    px, sline = make_spectrum()
    fitter = make_fitter()
    r2 = fitter.fit(px, sline, is_reference_mode=False)
    r4 = fitter.fit(px, sline, is_reference_mode=False, n_peaks=4)
    assert abs(r2.left_peak_center_px - r4.left_peak_center_px) < 0.03
    assert abs(r2.right_peak_center_px - r4.right_peak_center_px) < 0.03


def test_four_peaks_on_main_pair_only_roi_fails_loudly():
    # a spectrum holding only the main pair cannot support a 4-peak fit:
    # no silent fallback to a different layout
    px = np.arange(0.0, 200.0)
    true = np.full_like(px, OFFSET)
    for a, c, tau in ((3000.0, 83.0, TAU_L), (3000.0, 118.0, TAU_R)):
        true = true + psf_profile(px, a, c, GAMMA, SIGMA, tau)
    rng = np.random.default_rng(1)
    sline = true + rng.normal(0.0, 2.0, size=true.shape)
    result = make_fitter().fit(px, sline, is_reference_mode=False, n_peaks=4)
    assert not result.is_success


def test_four_peak_wrong_outer_tau_biases_outer_centre():
    # sanity that the per-position tails matter: fitting with the outer
    # tails swapped moves the outer centres by the tail convention
    fitter = make_fitter()
    fitter.psf_config = PsfConstants(
        psf_sigma_px=SIGMA, psf_tau_left_px=TAU_L, psf_tau_right_px=TAU_R,
        psf_tau_outer_left_px=TAU_OR, psf_tau_outer_right_px=TAU_OL)
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False, n_peaks=4)
    assert result.is_success
    assert abs(result.outer_left_peak_center_px - CENTERS[0]) > 0.15


# ---------------- outer-order tracks + the four-peak combination ----------


def _reference_fitter() -> SpectrumFitter:
    """A fitter whose sline is one frame row, for synthetic calibration
    frames (frame = the spectrum stacked over 3 rows)."""
    fitter = make_fitter()
    fitter.update_sline_config(SlineFromFrameConfig(
        pixel_offset_left=0, pixel_offset_right=0,
        selected_rows=[0, 1, 2], row_selection="manual"))
    return fitter


def _calibration_data(freqs_ghz, px_per_ghz=8.0):
    """Synthetic sweep: all four sideband positions move linearly with the
    set frequency, mimicking the real tracks."""
    blocks = []
    for i, f in enumerate(freqs_ghz):
        # each order's position moves with frequency; outer orders mirror
        # the inner slope sign like the real geometry (left tracks move
        # down-pixel with frequency, right tracks up-pixel)
        shift = px_per_ghz * (f - freqs_ghz[0])
        centers = (CENTERS[0] - shift, CENTERS[1] - shift,
                   CENTERS[2] + shift, CENTERS[3] + shift)
        _, sline = make_spectrum(seed=10 + i, centers=centers)
        frame = np.tile(sline / 3.0, (3, 1))
        blocks.append(MeasurementsPerFreq(
            set_freq_ghz=f,
            cali_meas_points=[CalibrationMeasurementPoint(
                frame=frame, microwave_freq=f)]))
    return CalibrationData(measured_freqs=blocks)


def test_outer_tracks_recover_the_synthetic_dispersion():
    freqs = [4.0, 4.5, 5.0, 5.5, 6.0]
    data = _calibration_data(freqs)
    tracks = build_outer_order_tracks(data, polyfit_degree=1,
                                      fitter=_reference_fitter())
    # slope: 1 GHz per 8 px, sign per side
    assert abs(tracks.dfreq_dpx_outer_left(30.0) - (-1.0 / 8.0)) < 0.01
    assert abs(tracks.dfreq_dpx_outer_right(150.0) - (1.0 / 8.0)) < 0.01
    # the track evaluates back to the set frequency at the fitted positions
    assert abs(float(tracks.freq_outer_left_ghz(
        tracks.outer_left_px_points[-1]))
        - tracks.outer_left_freq_points[-1]) < 0.02


def test_outer_tracks_refuse_a_main_pair_only_calibration():
    # frames holding only the main pair: the guard must raise, not fit junk
    freqs = [4.0, 5.0, 6.0]
    blocks = []
    px = np.arange(0.0, 200.0)
    for i, f in enumerate(freqs):
        true = np.full_like(px, OFFSET)
        for a, c, tau in ((3000.0, 83.0, TAU_L), (3000.0, 118.0, TAU_R)):
            true = true + psf_profile(px, a, c, GAMMA, SIGMA, tau)
        rng = np.random.default_rng(20 + i)
        sline = true + rng.normal(0.0, 2.0, size=true.shape)
        frame = np.tile(sline / 3.0, (3, 1))
        blocks.append(MeasurementsPerFreq(
            set_freq_ghz=f,
            cali_meas_points=[CalibrationMeasurementPoint(
                frame=frame, microwave_freq=f)]))
    with pytest.raises(ValueError, match="outer"):
        build_outer_order_tracks(CalibrationData(measured_freqs=blocks),
                                 polyfit_degree=1,
                                 fitter=_reference_fitter())


def test_four_peak_shift_combines_the_orders():
    freqs = [4.0, 4.5, 5.0, 5.5, 6.0]
    data = _calibration_data(freqs)
    fitter = _reference_fitter()
    tracks = build_outer_order_tracks(data, polyfit_degree=1, fitter=fitter)

    # inner-pair calculator from the same synthetic sweep geometry
    inner_left_px = [CENTERS[1] - 8.0 * (f - freqs[0]) for f in freqs]
    inner_right_px = [CENTERS[2] + 8.0 * (f - freqs[0]) for f in freqs]
    calc = CalibrationCalculator(CalibrationPolyfitParameters(
        degree=1,
        freq_left_peak=np.polyfit(inner_left_px, freqs, 1),
        freq_right_peak=np.polyfit(inner_right_px, freqs, 1),
        freq_peak_distance=np.polyfit(
            np.subtract(inner_right_px, inner_left_px), freqs, 1),
    ))

    # a "sample" frame at 5.0 GHz: all four orders agree by construction
    shift = 8.0 * (5.0 - freqs[0])
    centers = (CENTERS[0] - shift, CENTERS[1] - shift,
               CENTERS[2] + shift, CENTERS[3] + shift)
    px, sline = make_spectrum(seed=99, centers=centers)
    fs = fitter.fit(px, sline, is_reference_mode=False, n_peaks=4)
    assert fs.is_success

    result = four_peak_shift(fs, calc, tracks)
    # every order and the combination agree with the truth
    for f in result.freqs_ghz:
        assert abs(f - 5.0) < 0.02
    assert abs(result.combined_ghz - 5.0) < 0.02
    # weights: normalised, and the brighter inner pair dominates
    assert abs(sum(result.weights) - 1.0) < 1e-9
    assert result.weights[1] > result.weights[0]
    assert result.weights[2] > result.weights[3]


def test_four_peak_shift_requires_a_four_peak_fit():
    px, sline = make_spectrum()
    fs = make_fitter().fit(px, sline, is_reference_mode=False)  # two-peak
    with pytest.raises(ValueError, match="n_peaks=4"):
        four_peak_shift(fs, None, None)
