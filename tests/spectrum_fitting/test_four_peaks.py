"""Tests for the 4-peak fit (config n_peaks, or the fit(n_peaks=...)
override): selection by amplitude ranking, per-position tails
(psf_tau_outer_* for the outer orders), reported left/right = the inner
main pair, the outer_* result fields, the per-order calibration tracks
built by calibrate() in one pass, and the combined estimator.
"""
import numpy as np
import pytest

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationData,
    CalibrationMeasurementPoint,
    MeasurementsPerFreq,
    calibrate,
)
from dataclasses import replace

from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
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


def make_fitter(model="prm0", n_peaks=2) -> SpectrumFitter:
    fitter = SpectrumFitter()
    # kernel working values AND n_peaks live in the [global] sline config
    fitter.update_sline_config(replace(
        fitter.sline_config, n_peaks=n_peaks,
        psf_sigma_left_px=SIGMA, psf_sigma_right_px=SIGMA,
        psf_sigma_outer_left_px=SIGMA, psf_sigma_outer_right_px=SIGMA,
        psf_tau_left_px=TAU_L, psf_tau_right_px=TAU_R,
        psf_tau_outer_left_px=TAU_OL, psf_tau_outer_right_px=TAU_OR,
        psf_sat_ratio_outer_right=0.0, psf_sat_delta_outer_right_px=0.0))
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


def test_outer_right_satellite_removes_center_bias():
    # the outer_right order carries an intrinsic near-core satellite
    # (2026-09-02 determination): a scaled displaced copy of the main
    # line. A spectrum synthesized WITH the satellite must be fitted
    # without centre bias when the config carries the satellite
    # constants, and with a visible pull when it does not.
    sat_r, sat_d = 0.08, -1.23      # exaggerated ratio for a crisp test
    px = np.arange(0.0, 200.0)
    true = np.full_like(px, OFFSET)
    for a, c, tau in zip(AMPS, CENTERS, TAUS):
        true = true + psf_profile(px, a, c, GAMMA, SIGMA, tau)
    # add the satellite to the outer_right line only
    true = true + psf_profile(px, AMPS[3] * sat_r, CENTERS[3] + sat_d,
                              GAMMA, SIGMA, TAUS[3])
    rng = np.random.default_rng(7)
    sline = true + rng.normal(0.0, 2.0, size=true.shape)

    def fit_with(ratio, delta):
        fitter = make_fitter(n_peaks=4)
        fitter.update_sline_config(replace(
            fitter.sline_config,
            psf_sat_ratio_outer_right=ratio,
            psf_sat_delta_outer_right_px=delta))
        r = fitter.fit(px, sline, is_reference_mode=True, n_peaks=4)
        assert r.is_success
        return r

    with_sat = fit_with(sat_r, sat_d)
    without = fit_with(0.0, 0.0)
    err_with = abs(with_sat.outer_right_peak_center_px - CENTERS[3])
    err_without = abs(without.outer_right_peak_center_px - CENTERS[3])
    assert err_with < 0.03
    assert err_without > 2.0 * err_with
    # the satellite term must not disturb the other orders
    assert abs(with_sat.left_peak_center_px - CENTERS[1]) < 0.03
    assert abs(with_sat.right_peak_center_px - CENTERS[2]) < 0.03


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
    fitter.update_sline_config(replace(
        fitter.sline_config,
        psf_sigma_left_px=SIGMA, psf_sigma_right_px=SIGMA,
        psf_tau_left_px=TAU_L, psf_tau_right_px=TAU_R,
        psf_tau_outer_left_px=TAU_OR, psf_tau_outer_right_px=TAU_OL))
    px, sline = make_spectrum()
    result = fitter.fit(px, sline, is_reference_mode=False, n_peaks=4)
    assert result.is_success
    assert abs(result.outer_left_peak_center_px - CENTERS[0]) > 0.15


def test_global_config_drives_n_peaks():
    """n_peaks=4 in the GLOBAL sline config makes fit() four-peak without
    the argument — for sample and reference fits alike (one ROI, one
    peak count)."""
    fitter = make_fitter(n_peaks=4)
    px, sline = make_spectrum()
    for is_reference in (False, True):
        result = fitter.fit(px, sline, is_reference_mode=is_reference)
        assert result.is_success
        assert result.model.startswith("4")
        assert result.outer_left_peak_center_px is not None


def test_global_config_n_peaks_validation():
    from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
        SlineFromFrameConfig,
    )
    with pytest.raises(ValueError, match="n_peaks"):
        SlineFromFrameConfig(pixel_offset_left=0, pixel_offset_right=0,
                             selected_rows=[0], n_peaks=3)


# ---------------- per-order tracks + the four-peak combination ----------


def _reference_fitter(n_peaks=4) -> SpectrumFitter:
    """A fitter whose sline is one frame row, for synthetic calibration
    frames (frame = the spectrum stacked over 3 rows)."""
    fitter = make_fitter(n_peaks=n_peaks)
    fitter.update_sline_config(replace(
        fitter.sline_config,
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


def test_four_peak_calibration_builds_a_track_per_order():
    freqs = [4.0, 4.5, 5.0, 5.5, 6.0]
    data = _calibration_data(freqs)
    calc = CalibrationCalculator(calibrate(
        data, polyfit_degree=1, fitter=_reference_fitter()))
    assert calc.has_outer_tracks()
    # slope: 1 GHz per 8 px, sign per side
    assert abs(calc.dfreq_dpx_outer_left_peak(30.0) - (-1.0 / 8.0)) < 0.01
    assert abs(calc.dfreq_dpx_outer_right_peak(150.0) - (1.0 / 8.0)) < 0.01
    # the track evaluates back to the set frequency at the fitted positions
    assert abs(float(calc.freq_outer_left_peak(
        calc.p.outer_left_px_points[-1]))
        - calc.p.outer_left_freq_points[-1]) < 0.02
    # the inner tracks are built from the SAME four-peak fitting pass
    assert abs(calc.dfreq_dpx_left_peak(80.0) - (-1.0 / 8.0)) < 0.01


def test_two_peak_calibration_carries_no_outer_tracks():
    freqs = [4.0, 5.0, 6.0]
    data = _calibration_data(freqs)
    calc = CalibrationCalculator(calibrate(
        data, polyfit_degree=1, fitter=_reference_fitter(n_peaks=2)))
    assert not calc.has_outer_tracks()
    assert calc.p.freq_outer_left_peak is None


def test_four_peak_calibration_refuses_a_main_pair_only_roi():
    # frames holding only the main pair: every 4-peak fit fails, and the
    # raise names the n_peaks=4 requirement instead of fitting junk
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
    with pytest.raises(ValueError, match="n_peaks=4"):
        calibrate(CalibrationData(measured_freqs=blocks),
                  polyfit_degree=1, fitter=_reference_fitter())


def test_combined_shift_combines_the_orders():
    freqs = [4.0, 4.5, 5.0, 5.5, 6.0]
    data = _calibration_data(freqs)
    fitter = _reference_fitter()
    # ONE calibration pass yields the inner AND outer tracks
    calc = CalibrationCalculator(calibrate(data, polyfit_degree=1,
                                           fitter=fitter))

    # a "sample" frame at 5.0 GHz: all four orders agree by construction
    shift = 8.0 * (5.0 - freqs[0])
    centers = (CENTERS[0] - shift, CENTERS[1] - shift,
               CENTERS[2] + shift, CENTERS[3] + shift)
    px, sline = make_spectrum(seed=99, centers=centers)
    fs = fitter.fit(px, sline, is_reference_mode=False, n_peaks=4)
    assert fs.is_success

    result = calc.combined_shift(fs)
    # every order and the combination agree with the truth
    for f in result.freqs_ghz:
        assert abs(f - 5.0) < 0.02
    assert abs(result.combined_ghz - 5.0) < 0.02
    # weights: normalised, and the brighter inner pair dominates
    assert abs(sum(result.weights) - 1.0) < 1e-9
    assert result.weights[1] > result.weights[0]
    assert result.weights[2] > result.weights[3]

    # analyze() carries the per-order shifts and the combination
    shifts = calc.analyze(fs)
    assert shifts.freq_shift_combined_ghz == pytest.approx(result.combined_ghz)
    assert abs(shifts.freq_shift_outer_left_peak_ghz - 5.0) < 0.02
    assert abs(shifts.freq_shift_outer_right_peak_ghz - 5.0) < 0.02


def test_four_peak_calibration_builds_outer_width_tracks():
    # the outer orders get their own instrument-width polynomials from the
    # same fitting pass, so they carry the full width chain (2026-09-02)
    freqs = [4.0, 4.5, 5.0, 5.5, 6.0]
    calc = CalibrationCalculator(calibrate(
        _calibration_data(freqs), polyfit_degree=1,
        fitter=_reference_fitter()))
    assert calc.p.calibration_width_outer_left_peak is not None
    assert calc.p.calibration_width_outer_right_peak is not None
    # the sidebands are GAMMA wide in px, so the instrument width at any
    # outer pixel is GAMMA * |local dispersion| = 1.0 * (1/8) GHz
    inst_l, inst_r = calc.instrument_hwhm_outer_ghz(30.0, 150.0)
    assert abs(inst_l - 1.0 / 8.0) < 0.01
    assert abs(inst_r - 1.0 / 8.0) < 0.01


def test_outer_widths_and_linewidth_through_analyze():
    freqs = [4.0, 4.5, 5.0, 5.5, 6.0]
    fitter = _reference_fitter()
    calc = CalibrationCalculator(calibrate(
        _calibration_data(freqs), polyfit_degree=1, fitter=fitter))

    # a "sample" whose peaks are exactly as wide as the calibration
    # sidebands: raw HWHM = instrument HWHM, sample linewidth ~ 0
    shift = 8.0 * (5.0 - freqs[0])
    centers = (CENTERS[0] - shift, CENTERS[1] - shift,
               CENTERS[2] + shift, CENTERS[3] + shift)
    px, sline = make_spectrum(seed=7, centers=centers)
    fs = fitter.fit(px, sline, is_reference_mode=False, n_peaks=4)
    assert fs.is_success

    shifts = calc.analyze(fs)
    for raw, inst, lw in (
            (shifts.hwhm_outer_left_peak_ghz,
             shifts.instrument_hwhm_outer_left_peak_ghz,
             shifts.linewidth_outer_left_peak_ghz),
            (shifts.hwhm_outer_right_peak_ghz,
             shifts.instrument_hwhm_outer_right_peak_ghz,
             shifts.linewidth_outer_right_peak_ghz)):
        assert abs(raw - 1.0 / 8.0) < 0.01     # GAMMA px * (1/8) GHz/px
        assert abs(inst - 1.0 / 8.0) < 0.01
        assert abs(lw) < 0.01                  # same width -> ~0 sample HWHM
    # the inner-pair width observables are untouched
    assert shifts.hwhm_left_peak_ghz is not None
    assert shifts.linewidth_left_peak_ghz is not None


def test_outer_widths_none_without_outer_width_model():
    # a two-peak calibration has neither outer tracks nor outer widths:
    # every outer width observable degrades to None, loudly nothing
    freqs = [4.0, 5.0, 6.0]
    calc = CalibrationCalculator(calibrate(
        _calibration_data(freqs), polyfit_degree=1,
        fitter=_reference_fitter(n_peaks=2)))
    px, sline = make_spectrum(seed=8)
    fs = make_fitter().fit(px, sline, is_reference_mode=False, n_peaks=4)
    assert fs.is_success

    shifts = calc.analyze(fs)
    assert shifts.hwhm_outer_left_peak_ghz is None
    assert shifts.instrument_hwhm_outer_left_peak_ghz is None
    assert shifts.linewidth_outer_left_peak_ghz is None
    assert shifts.linewidth_outer_right_peak_ghz is None


def test_combined_shift_is_none_without_four_peaks():
    freqs = [4.0, 4.5, 5.0, 5.5, 6.0]
    calc = CalibrationCalculator(calibrate(
        _calibration_data(freqs), polyfit_degree=1,
        fitter=_reference_fitter()))
    px, sline = make_spectrum()
    fs = make_fitter().fit(px, sline, is_reference_mode=False)  # two-peak
    assert calc.combined_shift(fs) is None
    shifts = calc.analyze(fs)
    assert shifts.freq_shift_combined_ghz is None
    assert shifts.freq_shift_outer_left_peak_ghz is None
    # the inner-pair observables are untouched by the missing combination
    assert shifts.freq_shift_peak_distance_ghz is not None
