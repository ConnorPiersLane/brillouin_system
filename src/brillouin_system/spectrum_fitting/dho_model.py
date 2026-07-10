from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass
class ElasticAnchors:
    """
    Calibration-derived inputs for the anchored DHO / PSF sample fits.

    rayleigh_left_px / rayleigh_right_px : pixel positions of the elastic
        (Rayleigh) lines bracketing the two Brillouin peaks (the left order
        sits left of the left peak, the right order right of the right peak).
        This is ALL the pure eq.-S2 "dho" model needs.

    psf_left / psf_right / psf_grid_step_px : empirical per-order instrument
        response (ePSF = optical PSF including the 1-px binning),
        reconstructed from the calibration sweep. Consumed by the PSF-aware
        models (dho_psf, lorentzian_psf). The profile is sampled on a uniform
        grid of spacing psf_grid_step_px, symmetric around its maximum at the
        center index, area-normalised.

    chain : which calibration chain produced these anchors — "lorentzian"
        (the main chain) or "psf" (the PSF-centered variant). Fits that
        consume the anchors are stamped with this value
        (FittedSpectrum.calibration_chain) so the analysis maps their pixels
        through the same chain.

    psf_variant : anchors of the PSF-centered sibling chain (rayleigh
        positions from the variant polynomials + the ePSFs, chain="psf"),
        mirroring CalibrationPolyfitParameters.psf_variant. None when the
        calibration has no PSF chain. The fitter picks the base anchors for
        the pure dho and the variant for the PSF-aware models.

    See CalibrationCalculator.elastic_anchors.
    """
    rayleigh_left_px: float
    rayleigh_right_px: float
    psf_left: np.ndarray = None
    psf_right: np.ndarray = None
    psf_grid_step_px: float = None
    chain: str = "lorentzian"
    psf_variant: "ElasticAnchors" = None


def dho_intensity(u, omega0, gamma):
    """
    Damped harmonic oscillator spectral density (eq. S2 in Bailey et al.,
    Sci. Adv. 6, eabc1937 (2020), supplementary materials):

        S(u) = omega0^2 * gamma / ((omega0^2 - u^2)^2 + (u * gamma)^2)

    u      = frequency offset from the elastic (Rayleigh) line, here in pixel units
    omega0 = resonance frequency (pixel units)
    gamma  = damping rate; for gamma << omega0 the peak HWHM approaches gamma / 2

    S(u) is even in u, so a single DHO produces both the Stokes and the
    anti-Stokes hump around its elastic line.
    """
    u2 = np.square(u)
    return omega0**2 * gamma / (np.square(omega0**2 - u2) + u2 * gamma**2)


def dho_peak_height(omega0, gamma):
    """Value of dho_intensity at its maximum (for unit amplitude)."""
    denom = gamma * max(omega0**2 - 0.25 * gamma**2, 1e-12)
    return omega0**2 / denom


# Sub-pixel sampling used to integrate the bare eq.-S2 profile over each
# 1-px camera bin (same role the arctan closed form plays for the Lorentzian).
_S2_OVERSAMPLE = 11


def _dho_s2_pixel_integrated(x, amp, rayleigh_px, omega, gamma, oversample=_S2_OVERSAMPLE):
    """
    Bare eq. S2 (no instrument model) anchored at rayleigh_px, integrated
    over each pixel. The fitted gamma is the TOTAL observed damping —
    material and instrument broadening mixed, exactly as the paper's direct
    S2 fit.
    """
    x = np.asarray(x, dtype=float)
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    u = (x[:, None] + sub[None, :]) - float(rayleigh_px)
    return amp * dho_intensity(u, omega, gamma).mean(axis=1)


def make_2dho_s2_anchored(rayleigh_left_px, rayleigh_right_px):
    """
    Build the two-peak PURE eq.-S2 model: one bare DHO per Rayleigh order,
    anchored at the fixed elastic-line positions, pixel-integrated. No
    instrument response — the only calibration input is the anchors, and the
    fitted gammas are TOTAL observed dampings. Deliberately does NOT use the
    calibrated reference-line widths: coupling a fixed IRF width into the
    anchored fit proved ill-conditioned on ~1-px peaks (2026-07-10 analysis),
    while the bare fit's inputs (anchors) are insensitive.

    Free parameters:
        [amp1, omega1, gamma1, amp2, omega2, gamma2, offset]

    omega1 / omega2 are the resonances in pixel units, measured from each
    peak's elastic line.
    """
    rL = float(rayleigh_left_px)
    rR = float(rayleigh_right_px)

    def _2dho_s2_anchored(x, amp1, omega1, gamma1, amp2, omega2, gamma2, offset):
        return (
            _dho_s2_pixel_integrated(x, amp1, rL, omega1, gamma1)
            + _dho_s2_pixel_integrated(x, amp2, rR, omega2, gamma2)
            + offset
        )

    return _2dho_s2_anchored




def epsf_grid(psf, grid_step_px):
    """Offset axis of a stored ePSF: symmetric, zero at the center index."""
    psf = np.asarray(psf, dtype=float)
    n = len(psf)
    return grid_step_px * (np.arange(n) - (n - 1) / 2.0)


def epsf_hwhm_px(psf, grid_step_px):
    """
    HWHM (pixels) of a stored ePSF profile via its half-maximum crossings.
    For a skewed profile this is the mean of the two half-widths.
    """
    psf = np.asarray(psf, dtype=float)
    u = epsf_grid(psf, grid_step_px)
    i = int(np.argmax(psf))
    half_level = psf[i] / 2.0

    left = u[0]
    for j in range(i, 0, -1):
        if psf[j - 1] <= half_level:
            frac = (psf[j] - half_level) / max(psf[j] - psf[j - 1], 1e-12)
            left = u[j] - frac * grid_step_px
            break

    right = u[-1]
    for j in range(i, len(psf) - 1):
        if psf[j + 1] <= half_level:
            frac = (psf[j] - half_level) / max(psf[j] - psf[j + 1], 1e-12)
            right = u[j] + frac * grid_step_px
            break

    return 0.5 * float(right - left)


def _epsf_convolved_sampled(x, core_func, kernel, grid_step_px):
    """
    Convolve a material lineshape with the measured ePSF and SAMPLE at pixel
    centers.

    core_func(fine_px_axis) evaluates the material profile on the fine grid.
    The ePSF already contains the 1-px binning (it was reconstructed from
    pixel-integrated samples), so no additional pixel integration is applied
    — doing so would double-count the pixelation.
    """
    x = np.asarray(x, dtype=float)
    dt = float(grid_step_px)
    margin = dt * (len(kernel) - 1) / 2.0

    lo = float(x.min()) - margin - 1.0
    hi = float(x.max()) + margin + 1.0
    n = int(np.ceil((hi - lo) / dt)) + 1
    fine = lo + dt * np.arange(n)

    conv = fftconvolve(core_func(fine), kernel / kernel.sum(), mode="same")
    return np.interp(x, fine, conv)


def lorentzian_core(u, hwhm):
    """Height-normalised Lorentzian, HWHM in pixels."""
    hwhm = max(float(hwhm), 1e-6)
    return hwhm**2 / (np.square(u) + hwhm**2)


def _dho_epsf_sampled(x, amp, rayleigh_px, omega, gamma_mat, kernel, grid_step_px):
    """Material DHO (anchored at rayleigh_px) convolved with the measured ePSF."""
    return _epsf_convolved_sampled(
        x,
        lambda fine: amp * dho_intensity(fine - rayleigh_px, omega, gamma_mat),
        kernel,
        grid_step_px,
    )


def make_2dho_epsf_anchored(
    rayleigh_left_px, rayleigh_right_px, psf_left, psf_right, grid_step_px
):
    """
    Build the two-peak anchored DHO model convolved with the MEASURED
    per-order instrument response (ePSF) from the calibration's PSF chain.
    Handles skewed (asymmetric) instrument responses that a Lorentzian IRF
    cannot represent.

    Free parameters (same layout as the Lorentzian-IRF variant):
        [amp1, omega1, gamma_mat1, amp2, omega2, gamma_mat2, offset]
    """
    rL = float(rayleigh_left_px)
    rR = float(rayleigh_right_px)
    kL = np.asarray(psf_left, dtype=float)
    kR = np.asarray(psf_right, dtype=float)
    dt = float(grid_step_px)

    def _2dho_epsf_anchored(x, amp1, omega1, gmat1, amp2, omega2, gmat2, offset):
        return (
            _dho_epsf_sampled(x, amp1, rL, omega1, gmat1, kL, dt)
            + _dho_epsf_sampled(x, amp2, rR, omega2, gmat2, kR, dt)
            + offset
        )

    return _2dho_epsf_anchored


def make_2lorentzian_epsf(psf_left, psf_right, grid_step_px):
    """
    Two Lorentzian peaks, each convolved with its order's MEASURED instrument
    response (ePSF). The forward-model counterpart of the classic
    2lorentzian: because the instrument shape is in the model, the fitted
    centers are unbiased under a non-Lorentzian / skewed instrument response
    (pairs consistently with the calibration's PSF chain), and the fitted
    gammas are the MATERIAL HWHMs (instrument deconvolved). Needs no elastic
    anchors — centers are free parameters.

    Parameter order (same layout as the classic two-peak Lorentzian):
        [amp1, cen1, gamma1, amp2, cen2, gamma2, offset]

    amp is the material-core peak height; the observed (convolved) height is
    lower, approximately amp * gamma / (gamma + instrument_hwhm).
    """
    kL = np.asarray(psf_left, dtype=float)
    kR = np.asarray(psf_right, dtype=float)
    dt = float(grid_step_px)

    def _2lorentzian_epsf(x, amp1, cen1, gamma1, amp2, cen2, gamma2, offset):
        return (
            _epsf_convolved_sampled(
                x, lambda fine: amp1 * lorentzian_core(fine - cen1, gamma1), kL, dt)
            + _epsf_convolved_sampled(
                x, lambda fine: amp2 * lorentzian_core(fine - cen2, gamma2), kR, dt)
            + offset
        )

    return _2lorentzian_epsf


