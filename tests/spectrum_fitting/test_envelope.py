"""Tests for the per-scan VIPA envelope (spectrum_fitting/envelope.py).

Synthetic four-line calibration frames with a known ln-envelope g(x): the
same-sideband area ratios must recover g'(x) at the line positions. The
Envelope class (2026-09-09) owns the pair samples, the polynomial fit, the
outputs the chain reads (slope, ln_envelope, __call__, factor), a CSV
round trip, a degree refit and a comparison between two envelopes; the
four-peak DHO fit can apply the full curve instead of the local slope
(envelope_apply = "curve").
"""
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from brillouin_system.spectrum_fitting.dho import DhoAxes, dho_profile
from brillouin_system.spectrum_fitting.envelope import (
    AREA_HALF_PX, DEG, Envelope, EnvelopeComparison, EnvelopeModel,
    envelope_from_calibration)
from brillouin_system.spectrum_fitting.measured_kernel import MeasuredKernel
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig, SlineFromFrameConfig)
from brillouin_system.spectrum_fitting.psf import DX, psf_profile
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

PX = np.arange(200, dtype=float)
FLOOR = 2600.0
N_ROWS, ROW_BAND = 27, slice(7, 20)
# real geometry: FSR 21.7 GHz; tracks px(f) for the four lines
FSR = 21.7
def x_outer_left(f): return 37.0 - (f - 5.07) / 0.197
def x_left(f): return 80.0 + (f - 5.07) / 0.279
def x_right(f): return 116.0 - (f - 5.07) / 0.350
def x_outer_right(f): return 143.5 + (f - 5.07) / 0.403
# true ln envelope: a parabola peaking at px 94, with a linear term
def g_true(x): return -0.0003 * (x - 94.0) ** 2 + 0.002 * (x - 94.0)
def gprime_true(x): return -0.0006 * (x - 94.0) + 0.002
POSITIONS = (37.0, 80.0, 116.0, 143.5)


@dataclass
class _Point:
    frame: np.ndarray
    microwave_freq: float


@dataclass
class _Block:
    set_freq_ghz: float
    cali_meas_points: list


@dataclass
class _Calibration:
    measured_freqs: list


def frame_from_sline(sline):
    frame = np.full((N_ROWS, PX.size), FLOOR / 13.0)
    frame[ROW_BAND, :] = (sline / 13.0)[None, :]
    return frame


def synthetic_calibration(rng, drive_rolloff=True):
    """Four lines per frame; the +f pair (left, outer_right) share one drive
    amplitude, the -f pair (outer_left, right) another; both roll off with
    f; every line is multiplied by the envelope exp(g(x))."""
    blocks = []
    for k in range(41):
        f = 4.0 + 0.1 * k
        a_plus = 60000.0 * (1.0 - 0.08 * (f - 4.0)) if drive_rolloff else 60000.0
        a_minus = 0.85 * a_plus
        s = np.full(PX.size, FLOOR)
        for x0, amp in ((x_outer_left(f), a_minus), (x_left(f), a_plus),
                        (x_right(f), a_minus), (x_outer_right(f), a_plus)):
            s = s + amp * np.exp(g_true(x0)) * psf_profile(PX, 1.0, x0, 0.43, 0.26, 0.3)
        s = s + rng.normal(0.0, np.sqrt(s / 3.89 + 1.37))
        blocks.append(_Block(f, [_Point(frame_from_sline(s), f)]))
    return _Calibration(blocks)


def make_fitter():
    f = SpectrumFitter()
    f.update_sline_config(replace(f.sline_config, n_peaks=4, row_selection="manual"))
    return f


@pytest.fixture(scope="module")
def env():
    rng = np.random.default_rng(7)
    return envelope_from_calibration(synthetic_calibration(rng), make_fitter())


def test_envelope_slopes_recovered_from_area_ratios(env):
    assert isinstance(env, Envelope)
    assert env.n_frames >= 30
    assert env.rms_ln < 0.06
    assert env.deg == DEG and env.area_half_px == AREA_HALF_PX and env.area_method == "sum"
    for x in POSITIONS:
        assert abs(env.slope(x) - gprime_true(x)) < 0.003, (x, env.slope(x), gprime_true(x))


def test_drive_rolloff_cancels():
    rng = np.random.default_rng(8)
    with_roll = envelope_from_calibration(synthetic_calibration(rng, True), make_fitter())
    without = envelope_from_calibration(synthetic_calibration(rng, False), make_fitter())
    for x in (80.0, 116.0):
        assert abs(with_roll.slope(x) - without.slope(x)) < 0.002


def test_class_outputs(env):
    # two pair samples per usable frame, both pairs present, residuals fitted
    s = env.samples
    assert s.size == 2 * env.n_frames
    assert set(np.unique(s["pair"])) == {"+f", "-f"}
    assert abs(s["residual"].mean()) < 0.01
    assert np.isclose(np.sqrt(np.mean(s["residual"] ** 2)), env.rms_ln, rtol=0.05)
    # coverage spans the outer tracks
    lo, hi = env.coverage
    assert lo < 30.0 and hi > 150.0
    # the envelope normalised at its reference position, and its shape
    assert np.isclose(env(100.0, x_ref=100.0), 1.0)
    x = np.array([40.0, 80.0, 116.0, 140.0])
    ratio = env(x, x_ref=94.0) / np.exp(g_true(x) - g_true(94.0))
    assert np.all(np.abs(np.log(ratio)) < 0.06)
    # the full-curve factor equals the local slope to first order and
    # carries the curvature the slope does not
    c = 143.5
    u = np.linspace(-7.0, 7.0, 15)
    f_curve = env.factor(c + u, c)
    f_slope = np.exp(env.slope(c) * u)
    assert np.allclose(np.log(f_curve)[np.abs(u) <= 1.0], np.log(f_slope)[np.abs(u) <= 1.0], atol=2e-3)
    assert np.max(np.abs(np.log(f_curve) - np.log(f_slope))) > 0.005
    # binned residual structure is available for diagnostics
    cen, mean, rms, n = env.residuals_vs_x(np.arange(20.0, 161.0, 20.0))
    assert n.sum() == s.size
    assert repr(env).startswith("Envelope(deg=4")


def test_save_and_load_round_trip(env, tmp_path):
    path = tmp_path / "envelope.csv"
    env.save(path, source="synthetic")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#") and "coefficients=" in text and "deg=4" in text
    back = Envelope.load(path)
    assert back.deg == env.deg and back.n_frames == env.n_frames
    assert back.area_method == env.area_method and back.area_half_px == env.area_half_px
    assert np.allclose(back.coefficients, env.coefficients, atol=1e-9)
    assert np.allclose(back.samples["ln_ratio"], env.samples["ln_ratio"])
    assert np.allclose(back.samples["x_a"], env.samples["x_a"])
    assert back.source == "synthetic"
    for x in POSITIONS:
        assert np.isclose(back.slope(x), env.slope(x), atol=1e-8)
    # readable natively: pandas skips the comment lines
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(path, comment="#")
    assert list(df.columns) == ["frame", "freq_ghz", "pair", "x_a", "x_b", "ln_ratio", "residual"]
    assert len(df) == env.samples.size


def test_refit_degree_and_compare(env):
    d2 = env.refit(2)
    assert d2.deg == 2 and d2.coefficients.size == 2
    assert np.array_equal(d2.samples["ln_ratio"], env.samples["ln_ratio"])
    # the truth is a parabola: degree 2 recovers the slopes as well
    for x in POSITIONS:
        assert abs(d2.slope(x) - gprime_true(x)) < 0.003
    d5 = env.refit(5)
    assert d5.coefficients.size == 5 and d5.rms_ln <= env.rms_ln + 1e-9
    same = env.compare(env, POSITIONS)
    assert isinstance(same, EnvelopeComparison)
    assert np.all(same.d_slope == 0.0) and np.all(same.shift_mhz == 0.0)
    cmp = env.compare(d2, POSITIONS, mhz_per_slope=200.0)
    assert np.allclose(cmp.shift_mhz, cmp.d_slope * 200.0)
    assert "outer_right" in cmp.report() and "pull MHz" in str(cmp)


def test_template_area_estimator_matches_the_sum():
    """The area from the unit-area ePSF profile of the same calibration
    gives the same slopes as the summed counts (both see the whole line)."""
    from brillouin_system.spectrum_fitting.epsf import Epsf
    rng = np.random.default_rng(9)
    cal = synthetic_calibration(rng)
    fitter = make_fitter()
    fitter.update_sline_config(replace(fitter.sline_config, envelope_source="config"))
    epsf = Epsf.from_calibration(cal, fitter, n_lines=4)
    by_sum = Envelope.from_calibration(cal, fitter)
    by_template = Envelope.from_calibration(cal, fitter, area_method="template",
                                            profiles=epsf)
    assert by_template.area_method == "template" and by_template.n_frames >= 25
    for x in POSITIONS:
        assert abs(by_template.slope(x) - gprime_true(x)) < 0.003
        assert abs(by_template.slope(x) - by_sum.slope(x)) < 0.002
    with pytest.raises(ValueError, match="profiles"):
        Envelope.from_calibration(cal, fitter, area_method="template")
    with pytest.raises(ValueError, match="area_method"):
        Envelope.from_calibration(cal, fitter, area_method="peak")


def test_former_names_still_construct(env):
    m = EnvelopeModel(coefficients=env.coefficients, n_frames=env.n_frames,
                      rms_ln=env.rms_ln, x_min=env.x_min, x_max=env.x_max)
    assert isinstance(m, Envelope)
    assert np.isclose(m.slope(80.0), env.slope(80.0))


def test_config_validates_envelope_source_and_apply():
    cfg = SlineFromFrameConfig(pixel_offset_left=0, pixel_offset_right=0,
                               selected_rows=list(range(7, 20)), envelope_source="measured")
    assert cfg.envelope_source == "measured" and cfg.envelope_apply == "slope"
    with pytest.raises(ValueError, match="envelope_source"):
        SlineFromFrameConfig(pixel_offset_left=0, pixel_offset_right=0,
                             selected_rows=list(range(7, 20)), envelope_source="guess")
    with pytest.raises(ValueError, match="envelope_apply"):
        SlineFromFrameConfig(pixel_offset_left=0, pixel_offset_right=0,
                             selected_rows=list(range(7, 20)), envelope_apply="taylor")


# ---------------------------------------------------------------- fitter
# four-peak DHO fit with the envelope from a node-table stand-in: the
# full curve (envelope_apply = "curve") recovers a peak sitting under a
# curved envelope, the local slope leaves the curvature in the width.
SIGMA, TAU = 0.25, 0.3
G_INST_PX = 0.40
CENTRES = (22.0, 55.0, 95.0, 128.0)
SLOPES = (-0.20, 0.32, -0.36, 0.42)          # GHz/px per track
POLYS = [np.array([s, 5.0 - s * c]) for s, c in zip(SLOPES, CENTRES)]
GAMMA_GHZ = 0.30
GRID = np.arange(-6.0, 6.0 + DX / 2, DX)
FPX = np.arange(0, 160, dtype=float)
# a curved envelope, steeper than the bench (slope -0.085/px at the
# outer_right line) so the +-4 px fit window sees the curvature
def g_fit(x): return -0.0008 * (x - 75.0) ** 2


def _unit_kernel():
    k = psf_profile(GRID, 1.0, 0.0, G_INST_PX, SIGMA, TAU)
    return k / (k.sum() * DX)


class _Profiles:
    """The fitter's view of a node table: names, kernel_at, envelope."""
    names = ("outer_left", "left", "right", "outer_right")

    def __init__(self, envelope):
        self.envelope = envelope

    def kernel_at(self, line, cen):
        return MeasuredKernel(u=GRID, k=_unit_kernel(), position_px=float(cen),
                              n_frames=14, g_median_px=G_INST_PX)

    def env_slope(self, line, x):
        return float(self.envelope.slope(x))


def _four_peak_fitter(envelope_apply):
    cfg = dict(prominence_fraction=0.05, min_peak_width=1, min_peak_height=50,
               rel_height=0.5, wlen_pixels=20)
    f = SpectrumFitter()
    f.update_sample_config(FindPeaksConfig(fitting_model="dho_x_psf", **cfg))
    f.update_reference_config(FindPeaksConfig(fitting_model="lorentzian_x_psf", **cfg))
    f.update_sline_config(replace(
        f.sline_config, n_peaks=4, row_selection="manual", envelope_apply=envelope_apply,
        psf_sigma_left_px=SIGMA, psf_sigma_right_px=SIGMA,
        psf_tau_left_px=TAU, psf_tau_right_px=TAU,
        psf_box_outer_left_px=0.0, psf_box_outer_right_px=0.0,
        psf_sat_ratio_outer_right=0.0))
    return f


def _four_peak_truth():
    s = np.full(FPX.size, 100.0)
    for c, p, sl in zip(CENTRES, POLYS, SLOPES):
        m = np.abs(FPX - c) <= 15.0
        line = dho_profile(FPX[m], 3000.0, c, GAMMA_GHZ / abs(sl), p, G_INST_PX, SIGMA, TAU,
                           kernel=MeasuredKernel(u=GRID, k=_unit_kernel(), position_px=c,
                                                 n_frames=14, g_median_px=G_INST_PX))
        s[m] += line * np.exp(g_fit(FPX[m]) - g_fit(c))
    return s


def _fit_four(envelope_apply, truth):
    fitter = _four_peak_fitter(envelope_apply)
    kern = _Profiles(truth).kernel_at(0, 0.0)
    axes = DhoAxes(freq_left_poly=POLYS[1], freq_right_poly=POLYS[2],
                   freq_outer_left_poly=POLYS[0], freq_outer_right_poly=POLYS[3],
                   instrument_width_left_poly=np.array([G_INST_PX]),
                   instrument_width_right_poly=np.array([G_INST_PX]),
                   instrument_width_outer_left_poly=np.array([G_INST_PX]),
                   instrument_width_outer_right_poly=np.array([G_INST_PX]),
                   kernel_left=kern, kernel_right=kern,
                   kernel_outer_left=kern, kernel_outer_right=kern,
                   profiles=_Profiles(truth))
    fit = fitter.fit(FPX, _four_peak_truth(), is_reference_mode=False, n_peaks=4,
                     dho_axes=axes)
    assert fit.is_success
    widths = np.array([fit.outer_left_peak_width_px, fit.left_peak_width_px,
                       fit.right_peak_width_px, fit.outer_right_peak_width_px])
    cens = np.array([fit.outer_left_peak_center_px, fit.left_peak_center_px,
                     fit.right_peak_center_px, fit.outer_right_peak_center_px])
    return cens, widths


def test_four_peak_dho_applies_the_envelope_curve_or_slope():
    # g(u) with u = (x - 100)/50: -0.0008 (x - 75)^2 = -0.0008 (50 u + 25)^2
    # = -2 u^2 - 2 u (- 0.5, the constant is dropped)
    truth = Envelope.from_coefficients(coefficients=[-2.0, -2.0, 0.0, 0.0])
    assert np.isclose(truth.slope(128.0), -0.0016 * (128.0 - 75.0))
    truth_w = np.array([GAMMA_GHZ / abs(sl) for sl in SLOPES])
    cen_c, w_c = _fit_four("curve", truth)
    cen_s, w_s = _fit_four("slope", truth)
    # the full curve is the exact model: centres and widths back to the truth
    assert np.all(np.abs(cen_c - np.array(CENTRES)) < 0.03)
    assert np.all(np.abs(w_c / truth_w - 1.0) < 0.02)
    # the local slope misses the curvature and is measurably worse
    err_c = np.max(np.abs(w_c / truth_w - 1.0))
    err_s = np.max(np.abs(w_s / truth_w - 1.0))
    assert err_s > err_c + 0.005, (err_s, err_c)
