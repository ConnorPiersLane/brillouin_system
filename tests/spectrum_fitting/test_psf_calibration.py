"""
End-to-end tests for the empirical-PSF calibration (centering="psf") and the
PSF-kernel DHO fit.

Synthetic instrument: linear dispersion per order, a SKEWED optical PSF for
the left order (the real system's failure mode) and a symmetric one for the
right order. The calibration sweep slides the sidebands across the detector
in sub-pixel steps, exactly like the real EOM sweep.
"""
import numpy as np
import pytest

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationData,
    CalibrationMeasurementPoint,
    MeasurementsPerFreq,
    _bootstrap_reference_fits,
    _build_polyfit_parameters,
    _reconstruct_epsf,
    _refit_entries_with_epsf,
    calibrate,
)
from brillouin_system.spectrum_fitting.dho_model import (
    dho_intensity,
    epsf_grid,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FindPeaksConfig
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


# --- Synthetic instrument ---------------------------------------------------

N_PX = 80
N_ROWS_SIGNAL = 13     # global config sums rows 5..17
A_LEFT = 0.24          # GHz / px
A_RIGHT = 0.245        # GHz / px (non-rational px step, like the real system)
R_LEFT = 1.0           # true elastic-line pixel, left order
R_RIGHT = 78.0         # true elastic-line pixel, right order
FREQS = np.round(np.arange(4.0, 7.0 + 1e-9, 0.1), 6)

AMP_CAL = 40000.0      # summed-sline peak area scale for reference lines
OFFSET_CAL = 100.0


def optical_psf_left(u):
    """Skewed: heavier tail toward the elastic line (left side)."""
    u = np.asarray(u, dtype=float)
    hw = np.where(u < 0, 0.9, 0.45)
    return 1.0 / (1.0 + (u / hw) ** 2)


def optical_psf_right(u):
    """Symmetric Lorentzian, HWHM 0.6 px."""
    u = np.asarray(u, dtype=float)
    return 1.0 / (1.0 + (u / 0.6) ** 2)


def pixel_integrated(profile_func, px, center, oversample=21):
    """Mean of the optical profile over each 1-px bin (what the camera records)."""
    px = np.asarray(px, dtype=float)
    sub = np.linspace(-0.5, 0.5, oversample)
    grid = px[:, None] + sub[None, :]
    return profile_func(grid - center).mean(axis=1)


def left_center(freq):
    return R_LEFT + freq / A_LEFT


def right_center(freq):
    return R_RIGHT - freq / A_RIGHT


def make_reference_sline(freq):
    px = np.arange(N_PX, dtype=float)
    sline = (
        AMP_CAL * 0.1 * pixel_integrated(optical_psf_left, px, left_center(freq))
        + AMP_CAL * 0.08 * pixel_integrated(optical_psf_right, px, right_center(freq))
        + OFFSET_CAL
    )
    return sline


def sline_to_frame(sline):
    """Frame whose row sum over the configured rows reproduces sline."""
    frame = np.zeros((18, N_PX), dtype=float)
    frame[5:18, :] = sline[None, :] / N_ROWS_SIGNAL
    return frame


def make_calibration_data() -> CalibrationData:
    blocks = []
    for f in FREQS:
        point = CalibrationMeasurementPoint(
            frame=sline_to_frame(make_reference_sline(f)),
            microwave_freq=float(f),
            fitting_results=None,
        )
        blocks.append(MeasurementsPerFreq(
            set_freq_ghz=float(f), state_mode=None, cali_meas_points=[point],
        ))
    return CalibrationData(measured_freqs=blocks)


@pytest.fixture(scope="module")
def cal_data():
    return make_calibration_data()


@pytest.fixture(scope="module")
def params_psf(cal_data):
    return calibrate(cal_data, poyfit_degree=2, centering="psf")


# --- Calibration-level tests ------------------------------------------------

def test_lorentzian_centering_unchanged(cal_data):
    params = calibrate(cal_data, poyfit_degree=2)
    assert params.psf_left is None
    assert params.psf_right is None
    assert np.all(np.isfinite(params.freq_left_peak))
    assert np.all(np.isfinite(params.freq_right_peak))


def test_unknown_centering_raises(cal_data):
    with pytest.raises(ValueError, match="centering"):
        calibrate(cal_data, poyfit_degree=2, centering="banana")


def test_psf_centering_reconstructs_skewed_epsf(params_psf):
    assert params_psf.psf_left is not None
    assert params_psf.psf_right is not None
    step = params_psf.psf_grid_step_px
    assert step is not None and step > 0

    psf_l = np.asarray(params_psf.psf_left)
    psf_r = np.asarray(params_psf.psf_right)
    u = epsf_grid(psf_l, step)

    # Maximum sits at the center bin by convention
    assert abs(u[int(np.argmax(psf_l))]) <= step
    assert abs(u[int(np.argmax(psf_r))]) <= step

    # The left ePSF must be visibly skewed toward negative u (heavier tail
    # toward the elastic line); the right one symmetric.
    centroid_l = float(np.sum(u * psf_l) / np.sum(psf_l))
    centroid_r = float(np.sum(u * psf_r) / np.sum(psf_r))
    assert centroid_l < -0.05
    assert abs(centroid_r) < 0.05

    # Shape agreement with the true pixel-convolved PSF (left order).
    def box(profile_func, uu):
        sub = np.linspace(-0.5, 0.5, 21)
        return profile_func(np.asarray(uu)[:, None] + sub[None, :]).mean(axis=1)

    true_l = box(optical_psf_left, u + (u[int(np.argmax(psf_l))]))
    # align maxima: shift true profile so its max lands at u=0
    i_true = int(np.argmax(box(optical_psf_left, u)))
    true_l = box(optical_psf_left, u + u[i_true])
    true_l = true_l / true_l.sum()
    # normalized L1 distance
    l1 = float(np.abs(psf_l / psf_l.sum() - true_l).sum())
    assert l1 < 0.12


def test_psf_centering_reduces_center_bias(cal_data):
    """The shifted-PSF refit must track the true line positions better than
    the bootstrap Lorentzian fits (whose skew response wiggles with pixel
    phase). Constant offsets are irrelevant (absorbed by the polynomials),
    so errors are compared after demeaning."""
    entries = _bootstrap_reference_fits(cal_data)
    params = _build_polyfit_parameters(entries, 2)
    psf_l = _reconstruct_epsf(entries, params.freq_left_peak, side="left")
    psf_r = _reconstruct_epsf(entries, params.freq_right_peak, side="right")
    assert psf_l is not None and psf_r is not None
    refit = _refit_entries_with_epsf(entries, psf_l, psf_r)

    true_l = np.array([left_center(e.freq) for e in entries])
    err_boot = np.array([e.left_center_px for e in entries]) - true_l
    err_psf = np.array([e.left_center_px for e in refit]) - true_l

    rms_boot = float(np.std(err_boot))
    rms_psf = float(np.std(err_psf))
    assert rms_psf < rms_boot


# --- DHO with the measured PSF kernel ---------------------------------------

OMEGA_L = 21.0   # material resonance, px from left elastic line
OMEGA_R = 20.0
GAMMA_MAT = 3.0  # material gamma (HWHM 1.5 px)
AMP_S = 60000.0
OFFSET_S = 120.0


def make_sample_sline(seed=0):
    """Material DHO convolved with the TRUE optical PSFs, pixel-integrated."""
    px = np.arange(N_PX, dtype=float)
    fine = np.arange(-10.0, N_PX + 10.0, 0.02)

    def one(psf_func, rayleigh, omega, sign):
        u = sign * (fine - rayleigh)
        dho = dho_intensity(u, omega, GAMMA_MAT)
        kern_u = np.arange(-8.0, 8.0 + 1e-9, 0.02)
        kern = psf_func(kern_u)
        kern = kern / kern.sum()
        conv = np.convolve(dho, kern, mode="same")
        # pixel integration
        out = np.empty_like(px)
        for k, p in enumerate(px):
            m = np.abs(fine - p) <= 0.5
            out[k] = conv[m].mean()
        return out

    sline = (
        AMP_S * 0.02 * one(optical_psf_left, R_LEFT, OMEGA_L, +1)
        + AMP_S * 0.016 * one(optical_psf_right, R_RIGHT, OMEGA_R, -1)
        + OFFSET_S
    )
    rng = np.random.default_rng(seed)
    return px, sline + rng.normal(0.0, np.sqrt(np.clip(sline, 1.0, None)) * 0.2)


def sample_config(model="dho_window"):
    return FindPeaksConfig(
        prominence_fraction=0.05,
        min_peak_width=1,
        min_peak_height=50,
        rel_height=0.5,
        wlen_pixels=20,
        fitting_model=model,
        beta=4.0,
    )


def test_dho_with_measured_psf_recovers_material_parameters(params_psf):
    anchors = CalibrationCalculator(params_psf).elastic_anchors()
    assert anchors is not None
    assert anchors.psf_left is not None  # PSF carried through

    px, sline = make_sample_sline()
    fitter = SpectrumFitter()
    fitter.update_sample_config(sample_config())
    result = fitter.fit(px, sline, is_reference_mode=False, anchors=anchors)

    assert result.is_success
    assert result.model == "2dho_anchored_psf_window"

    omega_left = result.left_peak_center_px - result.rayleigh_left_px
    omega_right = result.rayleigh_right_px - result.right_peak_center_px
    assert omega_left == pytest.approx(OMEGA_L, abs=0.3)
    assert omega_right == pytest.approx(OMEGA_R, abs=0.3)

    assert result.material_hwhm_left_px == pytest.approx(GAMMA_MAT / 2, rel=0.25)
    assert result.material_hwhm_right_px == pytest.approx(GAMMA_MAT / 2, rel=0.25)


# --- lorentzian_psf: Lorentzian core forward-modeled through the ePSF -------

LP_CEN_LEFT = 22.0
LP_CEN_RIGHT = 58.0
LP_HWHM_MAT = 1.5


def make_lorentzian_sample_sline(seed=1):
    """Material Lorentzians convolved with the TRUE optical PSFs, pixel-integrated."""
    px = np.arange(N_PX, dtype=float)
    fine = np.arange(-10.0, N_PX + 10.0, 0.02)

    def one(psf_func, cen):
        core = 1.0 / (1.0 + ((fine - cen) / LP_HWHM_MAT) ** 2)
        kern_u = np.arange(-8.0, 8.0 + 1e-9, 0.02)
        kern = psf_func(kern_u)
        kern = kern / kern.sum()
        conv = np.convolve(core, kern, mode="same")
        out = np.empty_like(px)
        for k, p in enumerate(px):
            m = np.abs(fine - p) <= 0.5
            out[k] = conv[m].mean()
        return out

    sline = (
        1200.0 * one(optical_psf_left, LP_CEN_LEFT)
        + 950.0 * one(optical_psf_right, LP_CEN_RIGHT)
        + OFFSET_S
    )
    rng = np.random.default_rng(seed)
    return px, sline + rng.normal(0.0, np.sqrt(np.clip(sline, 1.0, None)) * 0.2)


@pytest.mark.parametrize("model", ["lorentzian_psf", "lorentzian_psf_window"])
def test_lorentzian_psf_recovers_centers_and_material_width(params_psf, model):
    anchors = CalibrationCalculator(params_psf).elastic_anchors()
    calc = CalibrationCalculator(params_psf)
    px, sline = make_lorentzian_sample_sline()

    fitter = SpectrumFitter()
    fitter.update_sample_config(sample_config(model))
    result = fitter.fit(px, sline, is_reference_mode=False, anchors=anchors)

    assert result.is_success
    expected = "2lorentzian_psf_window" if model == "lorentzian_psf_window" else "2lorentzian_psf"
    assert result.model == expected

    # The kernel's zero convention (maximum of the boxed profile at u=0)
    # offsets ALL centers — sample AND calibration — by the same constant,
    # which cancels through the calibration mapping, exactly as in the real
    # pipeline. So the unbiasedness check must be done in FREQUENCY space:
    true_freq_left = A_LEFT * (LP_CEN_LEFT - R_LEFT)
    true_freq_right = A_RIGHT * (R_RIGHT - LP_CEN_RIGHT)
    fitted_freq_left = float(calc.freq_left_peak(result.left_peak_center_px))
    fitted_freq_right = float(calc.freq_right_peak(result.right_peak_center_px))
    assert fitted_freq_left == pytest.approx(true_freq_left, abs=0.02)    # 20 MHz
    assert fitted_freq_right == pytest.approx(true_freq_right, abs=0.02)  # 20 MHz
    # Pixel-level sanity (allows the constant convention offset)
    assert result.left_peak_center_px == pytest.approx(LP_CEN_LEFT, abs=0.4)
    assert result.right_peak_center_px == pytest.approx(LP_CEN_RIGHT, abs=0.4)

    # Material widths recovered (instrument deconvolved)
    assert result.material_hwhm_left_px == pytest.approx(LP_HWHM_MAT, rel=0.2)
    assert result.material_hwhm_right_px == pytest.approx(LP_HWHM_MAT, rel=0.2)
    # Total (observed) width larger than material
    assert result.left_peak_width_px > result.material_hwhm_left_px
    assert result.right_peak_width_px > result.material_hwhm_right_px


def test_lorentzian_psf_without_psf_raises(cal_data):
    params_classic = calibrate(cal_data, poyfit_degree=2)  # no PSFs
    anchors = CalibrationCalculator(params_classic).elastic_anchors()
    px, sline = make_lorentzian_sample_sline()

    fitter = SpectrumFitter()
    fitter.update_sample_config(sample_config("lorentzian_psf"))
    with pytest.raises(ValueError, match="lorentzian_psf"):
        fitter.fit(px, sline, is_reference_mode=False, anchors=anchors)
    with pytest.raises(ValueError, match="lorentzian_psf"):
        fitter.fit(px, sline, is_reference_mode=False, anchors=None)


def test_lorentzian_psf_one_peak_fails_gracefully(params_psf):
    anchors = CalibrationCalculator(params_psf).elastic_anchors()
    px = np.arange(N_PX, dtype=float)
    sline = 1000.0 * np.exp(-0.5 * ((px - 40.0) / 2.0) ** 2) + 100.0

    fitter = SpectrumFitter()
    fitter.update_sample_config(sample_config("lorentzian_psf"))
    result = fitter.fit(px, sline, is_reference_mode=False, anchors=anchors)
    assert not result.is_success


def test_dho_without_psf_still_uses_lorentzian_irf(params_psf, cal_data):
    """A classic calibration (no PSF) keeps the Lorentzian-IRF DHO path."""
    params_classic = calibrate(cal_data, poyfit_degree=2)
    anchors = CalibrationCalculator(params_classic).elastic_anchors()
    assert anchors is not None
    assert anchors.psf_left is None

    px, sline = make_sample_sline()
    fitter = SpectrumFitter()
    fitter.update_sample_config(sample_config())
    result = fitter.fit(px, sline, is_reference_mode=False, anchors=anchors)

    assert result.is_success
    assert result.model == "2dho_anchored_window"
