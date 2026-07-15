"""Wiring tests for the na_lorentzian / na_lorentzian_window fitting models:
elastic-anchor extraction from calibration polynomials, model selection in
SpectrumFitter, and the raise-don't-fallback paths. The lineshape physics
itself is covered by test_na_lineshape.py.
"""
import numpy as np
import pytest

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationPolyfitParameters,
)
from brillouin_system.spectrum_fitting.elastic_anchors import ElasticAnchors
from brillouin_system.spectrum_fitting.na_correction5 import gaussian_angle_width
from brillouin_system.spectrum_fitting.na_lineshape import make_2na_lorentzian_binned
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FindPeaksConfig
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
    model_requires_anchors,
)


# ---- synthetic two-peak geometry (px axis) ----
R_LEFT = 5.0
R_RIGHT = 55.0
CEN_LEFT = 26.0    # true 180-degree positions
CEN_RIGHT = 34.0
GAMMA = 1.2
AMP = 1000.0
OFFSET = 20.0
NA_EFF = 0.42
N_SAMPLE = 1.33
ALPHA = float(np.arcsin(NA_EFF / N_SAMPLE))

ANCHORS = ElasticAnchors(rayleigh_left_px=R_LEFT, rayleigh_right_px=R_RIGHT)


BEAM_D_MM = 6.0    # collection-fiber mode (1/e^2 diameter) at the pupil
FOCAL_MM = 10.0    # objective focal length
V0 = float(gaussian_angle_width(BEAM_D_MM, FOCAL_MM, N_SAMPLE))


def make_config(model: str, na_collection: float = NA_EFF,
                beam_d: float = BEAM_D_MM, focal: float = FOCAL_MM) -> FindPeaksConfig:
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=50,
        rel_height=0.5,
        wlen_pixels=20,
        fitting_model=model,
        beta=4.0,
        na_collection=na_collection,
        na_beam_diameter_mm=beam_d,
        na_focal_length_mm=focal,
        na_n_sample=N_SAMPLE,
    )


def make_fitter(model: str, na_collection: float = NA_EFF,
                beam_d: float = BEAM_D_MM, focal: float = FOCAL_MM) -> SpectrumFitter:
    fitter = SpectrumFitter()
    fitter.update_sample_config(make_config(model, na_collection, beam_d, focal))
    return fitter


def make_spectrum(seed=0, v0=None):
    px = np.arange(0, 61, dtype=float)
    model = make_2na_lorentzian_binned(R_LEFT, R_RIGHT, ALPHA, n_quad=61, v0=v0)
    true = model(px, AMP, CEN_LEFT, GAMMA, AMP, CEN_RIGHT, GAMMA, OFFSET)
    rng = np.random.default_rng(seed)
    data = true + rng.normal(0.0, 1.0, size=true.shape)
    return px, data


# ---------------- elastic anchors from calibration ----------------

def _linear_params() -> CalibrationPolyfitParameters:
    # left sideband: freq grows moving right, zero at R_LEFT;
    # right sideband: freq grows moving left, zero at R_RIGHT.
    left_px = np.array([21.0, 25.0, 29.0, 33.0])
    right_px = np.array([39.0, 35.0, 31.0, 27.0])
    return CalibrationPolyfitParameters(
        degree=1,
        freq_left_peak=np.array([0.25, -0.25 * R_LEFT]),
        freq_right_peak=np.array([-0.25, 0.25 * R_RIGHT]),
        left_px_points=left_px,
        left_freq_points=0.25 * (left_px - R_LEFT),
        right_px_points=right_px[::-1],
        right_freq_points=(-0.25 * (right_px - R_RIGHT))[::-1],
    )


def test_elastic_anchors_linear():
    anchors = CalibrationCalculator(_linear_params()).elastic_anchors()
    assert anchors.rayleigh_left_px == pytest.approx(R_LEFT, abs=1e-8)
    assert anchors.rayleigh_right_px == pytest.approx(R_RIGHT, abs=1e-8)


def test_elastic_anchors_quadratic_newton():
    # degree-2 polys with a small curvature; Newton seeded from the sideband
    # points must still land on the nu=0 root.
    def poly_from(root, slope, curv):
        # freq(px) = slope*(px-root) + curv*(px-root)^2, expanded to coeffs
        return np.array([curv, slope - 2 * curv * root, curv * root**2 - slope * root])

    p = _linear_params()
    p.degree = 2
    p.freq_left_peak = poly_from(R_LEFT, 0.25, 1e-3)
    p.freq_right_peak = poly_from(R_RIGHT, -0.25, -1e-3)
    p.left_freq_points = np.polyval(p.freq_left_peak, p.left_px_points)
    p.right_freq_points = np.polyval(p.freq_right_peak, p.right_px_points)

    anchors = CalibrationCalculator(p).elastic_anchors()
    assert anchors.rayleigh_left_px == pytest.approx(R_LEFT, abs=1e-6)
    assert anchors.rayleigh_right_px == pytest.approx(R_RIGHT, abs=1e-6)


def test_elastic_anchors_raise_without_parameters():
    with pytest.raises(ValueError, match="no calibration parameters"):
        CalibrationCalculator(None).elastic_anchors()

    p = _linear_params()
    p.freq_left_peak = None
    with pytest.raises(ValueError, match="left elastic anchor"):
        CalibrationCalculator(p).elastic_anchors()

    p2 = _linear_params()
    p2.freq_right_peak = np.array([np.nan, np.nan])
    with pytest.raises(ValueError, match="right elastic anchor"):
        CalibrationCalculator(p2).elastic_anchors()


# ---------------- SpectrumFitter wiring ----------------

def test_model_requires_anchors():
    assert model_requires_anchors("na_lorentzian")
    assert model_requires_anchors("na_lorentzian_window")
    assert model_requires_anchors("na_gauss_lorentzian")
    assert model_requires_anchors("na_gauss_lorentzian_window")
    assert not model_requires_anchors("lorentzian")
    assert not model_requires_anchors("lorentzian_window")


@pytest.mark.parametrize("model", ["na_lorentzian", "na_lorentzian_window"])
def test_na_fit_recovers_180_positions(model):
    px, data = make_spectrum()
    fs = make_fitter(model).fit(px, data, is_reference_mode=False, anchors=ANCHORS)

    assert fs.is_success
    assert fs.model.startswith("2na_lorentzian")
    assert fs.left_peak_center_px == pytest.approx(CEN_LEFT, abs=0.03)
    assert fs.right_peak_center_px == pytest.approx(CEN_RIGHT, abs=0.03)
    # the standard value chain: distance is derived from the same centers
    assert fs.inter_peak_distance == pytest.approx(
        fs.right_peak_center_px - fs.left_peak_center_px, abs=1e-9
    )


def test_plain_lorentzian_on_na_data_is_biased_toward_anchors():
    """Old model on NA-broadened data: each apparent peak is pulled toward its
    own Rayleigh line — the bias the NA model removes."""
    px, data = make_spectrum()
    fs = make_fitter("lorentzian_window").fit(px, data, is_reference_mode=False)

    assert fs.is_success  # old model works without anchors, as before
    assert fs.left_peak_center_px < CEN_LEFT - 0.1
    assert fs.right_peak_center_px > CEN_RIGHT + 0.1


def test_na_fit_raises_without_anchors():
    px, data = make_spectrum()
    with pytest.raises(ValueError, match="requires elastic anchors"):
        make_fitter("na_lorentzian_window").fit(px, data, is_reference_mode=False)


def test_na_fit_raises_without_effective_na():
    px, data = make_spectrum()
    with pytest.raises(ValueError, match="na_collection"):
        make_fitter("na_lorentzian_window", na_collection=0.0).fit(
            px, data, is_reference_mode=False, anchors=ANCHORS
        )


@pytest.mark.parametrize("model", ["na_gauss_lorentzian", "na_gauss_lorentzian_window"])
def test_na_gauss_fit_recovers_180_positions(model):
    px, data = make_spectrum(v0=V0)
    fs = make_fitter(model).fit(px, data, is_reference_mode=False, anchors=ANCHORS)

    assert fs.is_success
    assert fs.model.startswith("2na_gauss_lorentzian")
    assert fs.left_peak_center_px == pytest.approx(CEN_LEFT, abs=0.03)
    assert fs.right_peak_center_px == pytest.approx(CEN_RIGHT, abs=0.03)


def test_na_gauss_kernel_differs_from_uniform():
    """The apodization suppresses large angles -> smaller mean downshift, so a
    uniform-kernel fit to Gaussian-weighted data lands below the truth."""
    px, data = make_spectrum(v0=V0)
    fs_uniform = make_fitter("na_lorentzian_window").fit(
        px, data, is_reference_mode=False, anchors=ANCHORS
    )
    assert fs_uniform.is_success
    # uniform kernel expects a bigger downshift, so it over-places the center
    # relative to the Gaussian-weighted truth (left peak beyond CEN_LEFT).
    assert fs_uniform.left_peak_center_px > CEN_LEFT + 0.02


def test_na_gauss_fit_raises_without_coupling_geometry():
    px, data = make_spectrum(v0=V0)
    with pytest.raises(ValueError, match="na_beam_diameter_mm"):
        make_fitter("na_gauss_lorentzian_window", beam_d=0.0).fit(
            px, data, is_reference_mode=False, anchors=ANCHORS
        )
    with pytest.raises(ValueError, match="na_beam_diameter_mm"):
        make_fitter("na_gauss_lorentzian_window", focal=0.0).fit(
            px, data, is_reference_mode=False, anchors=ANCHORS
        )


def test_na_fit_single_peak_fails():
    px = np.arange(0, 61, dtype=float)
    model = make_2na_lorentzian_binned(R_LEFT, R_RIGHT, ALPHA, n_quad=61)
    data = model(px, AMP, CEN_LEFT, GAMMA, 0.0, CEN_RIGHT, GAMMA, OFFSET)
    fs = make_fitter("na_lorentzian_window").fit(px, data, is_reference_mode=False, anchors=ANCHORS)
    assert not fs.is_success
