import numpy as np
import pytest

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationPolyfitParameters,
)
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.dho_model import (
    ElasticAnchors,
    dho_intensity,
    make_2dho_s2_anchored,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FindPeaksConfig
from brillouin_system.spectrum_fitting.spectrum_analyzer import SpectrumAnalyzer
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


# Ground truth for the synthetic spectrum. Rayleigh anchors sit outside the
# fitted pixel window so only the inner Stokes / anti-Stokes pair is visible,
# as on the real spectrometer. The pure dho fits bare eq. S2, so the
# synthetic data is the bare lineshape; GAMMA is the TOTAL damping.
RAYLEIGH_LEFT = 120.0
RAYLEIGH_RIGHT = 392.0
OMEGA0 = 60.0
GAMMA = 8.0           # total gamma -> observed HWHM = 4.0
GAMMA_INST = 2.0      # instrument HWHM (px); only used by the width polys
AMP_LEFT = 8000.0
AMP_RIGHT = 6000.0
OFFSET = 50.0

INST_WIDTH_POLY = np.array([GAMMA_INST])  # constant instrument HWHM vs px


def make_anchors(r_left, r_right) -> ElasticAnchors:
    return ElasticAnchors(rayleigh_left_px=r_left, rayleigh_right_px=r_right)


TRUE_ANCHORS = make_anchors(RAYLEIGH_LEFT, RAYLEIGH_RIGHT)


def make_config(model: str) -> FindPeaksConfig:
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=100,
        rel_height=0.5,
        wlen_pixels=40,
        fitting_model=model,
        beta=4.0,
    )


def make_synthetic_dho_sline(seed=0, gamma_left=GAMMA, gamma_right=GAMMA):
    """Bare eq.-S2 pair (pixel-integrated) with photon-like noise."""
    px = np.arange(150, 380, dtype=float)
    model = make_2dho_s2_anchored(RAYLEIGH_LEFT, RAYLEIGH_RIGHT)
    signal = model(px, AMP_LEFT, OMEGA0, gamma_left, AMP_RIGHT, OMEGA0, gamma_right, OFFSET)
    rng = np.random.default_rng(seed)
    noisy = signal + rng.normal(0.0, np.sqrt(np.clip(signal, 1.0, None)))
    return px, noisy


def fit_with_model(model: str, px, sline, anchors=None):
    fitter = SpectrumFitter()
    fitter.update_sample_config(make_config(model))
    return fitter.fit(px, sline, is_reference_mode=False, anchors=anchors)


def test_dho_without_anchors_raises():
    """The DHO model is eq. S2 anchored at the elastic lines; without the
    calibration-derived anchors it must refuse to fit rather than silently
    degrade."""
    px, sline = make_synthetic_dho_sline()

    with pytest.raises(ValueError, match="anchors"):
        fit_with_model("dho", px, sline)
    with pytest.raises(ValueError, match="anchors"):
        fit_with_model("dho_window", px, sline)


def test_dho_with_invalid_anchors_raises():
    px, sline = make_synthetic_dho_sline()
    with pytest.raises(ValueError, match="Invalid elastic anchors"):
        fit_with_model("dho", px, sline,
                       anchors=ElasticAnchors(rayleigh_left_px=400.0, rayleigh_right_px=100.0))
    with pytest.raises(ValueError, match="Invalid elastic anchors"):
        fit_with_model("dho", px, sline,
                       anchors=ElasticAnchors(rayleigh_left_px=np.nan, rayleigh_right_px=392.0))


def test_dho_psf_without_psf_raises():
    """dho_psf needs the measured ePSFs; anchors without them (classic
    calibration) must raise, never silently fall back."""
    px, sline = make_synthetic_dho_sline()
    with pytest.raises(ValueError, match="dho_psf"):
        fit_with_model("dho_psf", px, sline, anchors=TRUE_ANCHORS)
    with pytest.raises(ValueError, match="dho_psf"):
        fit_with_model("dho_psf_window", px, sline, anchors=TRUE_ANCHORS)


def test_dho_fit_fails_gracefully_with_one_peak():
    px = np.arange(0, 200, dtype=float)
    sline = 1000.0 * np.exp(-0.5 * ((px - 100.0) / 3.0) ** 2) + 20.0

    result = fit_with_model("dho", px, sline, anchors=make_anchors(20.0, 180.0))

    assert not result.is_success


def test_s2_pixel_integration_matches_bare_intensity():
    """The pixel-integrated pure-S2 model must reproduce the bare lineshape:
    same total area, and pointwise agreement away from sharp curvature."""
    px = np.arange(60, 452, dtype=float)
    model = make_2dho_s2_anchored(RAYLEIGH_LEFT, RAYLEIGH_RIGHT)
    binned = model(px, AMP_LEFT, OMEGA0, GAMMA, AMP_RIGHT, OMEGA0, GAMMA, 0.0)
    bare = (
        AMP_LEFT * dho_intensity(px - RAYLEIGH_LEFT, OMEGA0, GAMMA)
        + AMP_RIGHT * dho_intensity(px - RAYLEIGH_RIGHT, OMEGA0, GAMMA)
    )
    assert binned.sum() == pytest.approx(bare.sum(), rel=0.01)
    assert np.max(np.abs(binned - bare)) / bare.max() < 0.05


def test_dho_fit_recovers_two_different_widths():
    """The two peaks may have genuinely different linewidths; the pure S2
    fit must recover the two total gammas independently."""
    gamma_left = 6.0    # HWHM 3.0
    gamma_right = 12.0  # HWHM 6.0
    px, sline = make_synthetic_dho_sline(seed=3, gamma_left=gamma_left, gamma_right=gamma_right)

    result = fit_with_model("dho", px, sline, anchors=TRUE_ANCHORS)

    assert result.is_success
    assert result.model == "2dho_s2"
    # Total observed HWHM = gamma / 2, recovered independently per peak.
    assert result.left_peak_width_px == pytest.approx(gamma_left / 2, rel=0.1)
    assert result.right_peak_width_px == pytest.approx(gamma_right / 2, rel=0.1)
    assert result.right_peak_width_px > 1.4 * result.left_peak_width_px
    # Pure S2 has no instrument model, so no material/instrument split.
    assert result.material_hwhm_left_px is None
    assert result.material_hwhm_right_px is None
    # Centers are the resonance positions
    assert result.left_peak_center_px == pytest.approx(RAYLEIGH_LEFT + OMEGA0, abs=0.8)
    assert result.right_peak_center_px == pytest.approx(RAYLEIGH_RIGHT - OMEGA0, abs=0.8)


@pytest.mark.parametrize("model", ["dho", "dho_window"])
def test_pure_dho_recovers_resonance_and_total_width(model):
    """The pure eq.-S2 fit needs ONLY the anchor positions: the reported
    peak centers are the RESONANCE positions (rayleigh +/- omega) and the
    width fields carry the total observed HWHM (gamma / 2)."""
    px, sline = make_synthetic_dho_sline()

    result = fit_with_model(model, px, sline, anchors=TRUE_ANCHORS)

    assert result.is_success
    expected_kind = "2dho_s2_window" if model == "dho_window" else "2dho_s2"
    assert result.model == expected_kind
    assert result.calibration_chain == "lorentzian"

    assert result.rayleigh_left_px == RAYLEIGH_LEFT
    assert result.rayleigh_right_px == RAYLEIGH_RIGHT

    # Centers are the resonance positions: rayleigh +/- omega
    assert result.left_peak_center_px == pytest.approx(RAYLEIGH_LEFT + OMEGA0, abs=0.6)
    assert result.right_peak_center_px == pytest.approx(RAYLEIGH_RIGHT - OMEGA0, abs=0.6)

    omega_left = result.left_peak_center_px - result.rayleigh_left_px
    omega_right = result.rayleigh_right_px - result.right_peak_center_px
    assert omega_left == pytest.approx(OMEGA0, rel=0.02)
    assert omega_right == pytest.approx(OMEGA0, rel=0.02)

    # Total observed width; no material split without an instrument model.
    assert result.left_peak_width_px == pytest.approx(GAMMA / 2, rel=0.1)
    assert result.material_hwhm_left_px is None
    assert result.offset == pytest.approx(OFFSET, abs=15.0)


def test_non_bracketing_anchors_return_unsuccessful_fit():
    """Anchors that do not bracket the detected peaks (e.g. calibration
    drift) must not produce a fit — and must not crash a scan loop."""
    px, sline = make_synthetic_dho_sline()
    bad_anchors = make_anchors(250.0, 300.0)

    result = fit_with_model("dho", px, sline, anchors=bad_anchors)

    assert not result.is_success
    assert result.rayleigh_left_px is None


def test_other_models_ignore_anchors():
    px, sline = make_synthetic_dho_sline()
    result = fit_with_model("lorentzian_window", px, sline, anchors=TRUE_ANCHORS)
    assert result.is_success
    assert result.rayleigh_left_px is None
    assert result.material_hwhm_left_px is None


# ---------------------------------------------------------------------------
# Elastic anchor extraction from the calibration
# ---------------------------------------------------------------------------

A_LEFT = 0.05    # GHz / px
A_RIGHT = -0.05  # GHz / px


def make_linear_calibration_params() -> CalibrationPolyfitParameters:
    """Linear dispersion: nu = a * (px - R) for each peak, with sideband
    points covering 4-8 GHz (anchors lie outside the measured range)."""
    left_freqs = np.linspace(4.0, 8.0, 9)
    left_px = RAYLEIGH_LEFT + left_freqs / A_LEFT
    right_freqs = np.linspace(8.0, 4.0, 9)
    right_px = RAYLEIGH_RIGHT + right_freqs / A_RIGHT

    return CalibrationPolyfitParameters(
        degree=1,
        freq_left_peak=np.polyfit(left_px, left_freqs, 1),
        freq_right_peak=np.polyfit(right_px, right_freqs, 1),
        freq_peak_distance=np.array([-0.025, 10.0]),
        calibration_width_left_peak=INST_WIDTH_POLY,
        calibration_width_right_peak=INST_WIDTH_POLY,
        left_px_points=left_px,
        left_freq_points=left_freqs,
        right_px_points=right_px,
        right_freq_points=right_freqs,
    )


def test_elastic_anchors_linear_calibration_exact():
    calc = CalibrationCalculator(make_linear_calibration_params())
    anchors = calc.elastic_anchors()

    assert anchors is not None
    assert anchors.rayleigh_left_px == pytest.approx(RAYLEIGH_LEFT, abs=1e-6)
    assert anchors.rayleigh_right_px == pytest.approx(RAYLEIGH_RIGHT, abs=1e-6)
    assert anchors.chain == "lorentzian"
    assert anchors.psf_variant is None


def test_elastic_anchors_quadratic_calibration_small_bias():
    """Newton step from the point nearest zero keeps the extrapolation bias
    of a mildly quadratic dispersion within ~2 px."""
    curvature = 1e-5  # GHz / px^2

    u = np.linspace(4.0, 8.0, 9) / A_LEFT  # px offsets from the elastic line
    left_px = RAYLEIGH_LEFT + u
    left_freqs = A_LEFT * u + curvature * u**2

    u_r = np.linspace(4.0, 8.0, 9) / (-A_RIGHT)
    right_px = RAYLEIGH_RIGHT - u_r
    right_freqs = -A_RIGHT * u_r + curvature * u_r**2

    params = CalibrationPolyfitParameters(
        degree=2,
        freq_left_peak=np.polyfit(left_px, left_freqs, 2),
        freq_right_peak=np.polyfit(right_px, right_freqs, 2),
        freq_peak_distance=np.array([-0.025, 10.0]),
        calibration_width_left_peak=INST_WIDTH_POLY,
        calibration_width_right_peak=INST_WIDTH_POLY,
        left_px_points=np.sort(left_px),
        left_freq_points=left_freqs,
        right_px_points=np.sort(right_px),
        right_freq_points=right_freqs[np.argsort(right_px)],
    )

    anchors = CalibrationCalculator(params).elastic_anchors()

    assert anchors is not None
    assert anchors.rayleigh_left_px == pytest.approx(RAYLEIGH_LEFT, abs=2.0)
    assert anchors.rayleigh_right_px == pytest.approx(RAYLEIGH_RIGHT, abs=2.0)


def test_elastic_anchors_nan_calibration_returns_none():
    params = make_linear_calibration_params()
    params.freq_left_peak = np.full(2, np.nan)
    assert CalibrationCalculator(params).elastic_anchors() is None


def test_elastic_anchors_served_without_width_polys():
    """The pure eq.-S2 dho needs only the anchor positions, so anchors must
    be served even when the calibration has no width polynomials."""
    params = make_linear_calibration_params()
    params.calibration_width_left_peak = None
    params.calibration_width_right_peak = None
    anchors = CalibrationCalculator(params).elastic_anchors()
    assert anchors is not None
    assert anchors.rayleigh_left_px == pytest.approx(RAYLEIGH_LEFT, abs=1e-6)


def test_elastic_anchors_degree2_without_points_returns_none():
    params = make_linear_calibration_params()
    params.freq_left_peak = np.array([1e-6, 0.05, -6.0])  # degree 2
    params.left_px_points = None
    params.left_freq_points = None
    assert CalibrationCalculator(params).elastic_anchors() is None


def test_freq_shifts_of_anchored_fit_are_true_brillouin_shifts():
    """For the anchored DHO the peak centers hold the resonance positions,
    so the standard freq_shift outputs directly carry the true shift, and
    material HWHM is exposed separately for the loss modulus."""
    calc = CalibrationCalculator(make_linear_calibration_params())
    analyzer = SpectrumAnalyzer(calibration_calculator=calc)

    fitting = FittedSpectrum(
        is_success=True,
        x_pixels=np.arange(10, dtype=float),
        sline=np.zeros(10),
        model="2dho_anchored_psf",
        left_peak_center_px=RAYLEIGH_LEFT + OMEGA0,
        left_peak_width_px=GAMMA / 2 + GAMMA_INST,
        right_peak_center_px=RAYLEIGH_RIGHT - OMEGA0,
        right_peak_width_px=GAMMA / 2 + GAMMA_INST,
        inter_peak_distance=(RAYLEIGH_RIGHT - OMEGA0) - (RAYLEIGH_LEFT + OMEGA0),
        rayleigh_left_px=RAYLEIGH_LEFT,
        rayleigh_right_px=RAYLEIGH_RIGHT,
        material_hwhm_left_px=GAMMA / 2,
        material_hwhm_right_px=GAMMA / 2,
    )

    shifts = analyzer.analyze_spectrum(fitting)

    assert shifts.freq_shift_left_peak_ghz == pytest.approx(A_LEFT * OMEGA0, rel=1e-9)
    assert shifts.freq_shift_right_peak_ghz == pytest.approx(abs(A_RIGHT) * OMEGA0, rel=1e-9)
    # Material HWHM in GHz (for the loss modulus)
    assert shifts.material_hwhm_left_ghz == pytest.approx(A_LEFT * GAMMA / 2, rel=1e-9)
    assert shifts.material_hwhm_right_ghz == pytest.approx(abs(A_RIGHT) * GAMMA / 2, rel=1e-9)


def test_lorentzian_window_regression_for_calibration_pipeline():
    """The calibration pipeline fits with lorentzian_window; make sure the
    DHO additions did not disturb that code path."""
    px = np.arange(0, 512, dtype=float)
    cen1, cen2, wid = 180.0, 330.0, 3.0
    signal = (
        2000.0 / (1.0 + ((px - cen1) / wid) ** 2)
        + 1500.0 / (1.0 + ((px - cen2) / wid) ** 2)
        + 30.0
    )
    rng = np.random.default_rng(1)
    sline = signal + rng.normal(0.0, np.sqrt(np.clip(signal, 1.0, None)))

    result = fit_with_model("lorentzian_window", px, sline)

    assert result.is_success
    assert result.model == "2lorentzian_window"
    assert result.left_peak_center_px == pytest.approx(cen1, abs=0.5)
    assert result.right_peak_center_px == pytest.approx(cen2, abs=0.5)
    assert result.left_peak_width_px == pytest.approx(wid, rel=0.2)
