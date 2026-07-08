from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass
class ElasticAnchors:
    """
    Calibration-derived inputs for the anchored DHO fit.

    rayleigh_left_px / rayleigh_right_px : pixel positions of the elastic
        (Rayleigh) lines bracketing the two Brillouin peaks (the left order
        sits left of the left peak, the right order right of the right peak).

    instrument_width_left_poly / instrument_width_right_poly : polynomial
        coefficients (np.polyval order) giving the instrument HWHM in pixels
        as a function of pixel position, i.e. the calibration_width_*_peak
        models. These define the Lorentzian IRF that the DHO model is
        convolved with, so the fitted gammas are the *material* widths.

    psf_left / psf_right / psf_grid_step_px : empirical per-order instrument
        response (ePSF = optical PSF including the 1-px binning),
        reconstructed by a calibration run with centering="psf". When
        present, the DHO uses the measured (possibly skewed) kernel instead
        of the analytic Lorentzian IRF. The profile is sampled on a uniform
        grid of spacing psf_grid_step_px, symmetric around its maximum at the
        center index, area-normalised.

    See CalibrationCalculator.elastic_anchors.
    """
    rayleigh_left_px: float
    rayleigh_right_px: float
    instrument_width_left_poly: np.ndarray = None
    instrument_width_right_poly: np.ndarray = None
    psf_left: np.ndarray = None
    psf_right: np.ndarray = None
    psf_grid_step_px: float = None


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


def lorentzian_irf(t, gamma_inst):
    """Area-normalised Lorentzian instrument response, HWHM gamma_inst (pixels).

    Lorentzian to match how the instrument width is measured: the calibration
    fits the reference sidebands (the physical IRF) with a Lorentzian, so the
    stored width IS a Lorentzian HWHM. In the sharp limit this convolution
    reduces to simple width addition.
    """
    gamma_inst = max(float(gamma_inst), 1e-6)
    return (1.0 / np.pi) * gamma_inst / (np.square(t) + gamma_inst**2)


# Convolution resolution: fine grid spacing = 1 / _CONV_OVERSAMPLE pixel; the
# IRF kernel spans +/- _CONV_KERNEL_HALF * gamma_inst. The Lorentzian is
# truncated there (a real VIPA IRF does not have infinite Lorentzian tails).
_CONV_OVERSAMPLE = 8
_CONV_KERNEL_HALF = 15.0


def _dho_conv_pixel_integrated(
    x, amp, rayleigh_px, omega, gamma_mat, gamma_inst,
    oversample=_CONV_OVERSAMPLE, kernel_half=_CONV_KERNEL_HALF,
):
    """
    Material DHO (anchored at rayleigh_px) convolved with the Lorentzian
    instrument response and integrated over each pixel.

    omega / gamma_mat are the material resonance and damping; gamma_inst is the
    fixed instrument HWHM. Convolution is evaluated numerically on a fine grid
    extended by the kernel span so window edges are not corrupted.
    """
    x = np.asarray(x, dtype=float)
    gamma_inst = max(float(gamma_inst), 1e-6)
    dt = 1.0 / oversample
    margin = kernel_half * gamma_inst

    lo = float(x.min()) - margin - 0.5
    hi = float(x.max()) + margin + 0.5
    n = int(np.ceil((hi - lo) / dt)) + 1
    fine = lo + dt * np.arange(n)

    dho_fine = amp * dho_intensity(fine - rayleigh_px, omega, gamma_mat)

    half = int(np.ceil(margin / dt))
    kt = dt * np.arange(-half, half + 1)
    kernel = lorentzian_irf(kt, gamma_inst)
    kernel = kernel / kernel.sum()

    conv = fftconvolve(dho_fine, kernel, mode="same")

    # Pixel-integrate: mean of the fine samples inside [xi - 0.5, xi + 0.5].
    csum = np.concatenate([[0.0], np.cumsum(conv)])
    li = np.searchsorted(fine, x - 0.5, side="left")
    ri = np.searchsorted(fine, x + 0.5, side="right")
    counts = np.maximum(ri - li, 1)
    return (csum[ri] - csum[li]) / counts


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


def _dho_epsf_sampled(x, amp, rayleigh_px, omega, gamma_mat, kernel, grid_step_px):
    """
    Material DHO (anchored at rayleigh_px) convolved with the measured ePSF
    and SAMPLED at pixel centers.

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

    dho_fine = amp * dho_intensity(fine - rayleigh_px, omega, gamma_mat)
    conv = fftconvolve(dho_fine, kernel / kernel.sum(), mode="same")
    return np.interp(x, fine, conv)


def make_2dho_epsf_anchored(
    rayleigh_left_px, rayleigh_right_px, psf_left, psf_right, grid_step_px
):
    """
    Build the two-peak anchored DHO model convolved with the MEASURED
    per-order instrument response (ePSF), e.g. from a calibration run with
    centering="psf". Handles skewed (asymmetric) instrument responses that a
    Lorentzian IRF cannot represent.

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


def make_2dho_conv_anchored(
    rayleigh_left_px, rayleigh_right_px, gamma_inst_left, gamma_inst_right
):
    """
    Build the two-peak anchored DHO model convolved with the instrument
    response (eq. S2 twice, each convolved with its own Lorentzian IRF).

    Anchors and instrument widths are fixed; the free parameters are the
    material amplitude, resonance and damping of each peak:

        [amp1, omega1, gamma_mat1, amp2, omega2, gamma_mat2, offset]

    omega1 / omega2 are the true Brillouin resonances (pixel units, measured
    from each peak's elastic line) and gamma_mat1 / gamma_mat2 the material
    dampings, both corrected for instrument broadening by the convolution.
    """
    rL = float(rayleigh_left_px)
    rR = float(rayleigh_right_px)
    giL = float(gamma_inst_left)
    giR = float(gamma_inst_right)

    def _2dho_conv_anchored(x, amp1, omega1, gmat1, amp2, omega2, gmat2, offset):
        return (
            _dho_conv_pixel_integrated(x, amp1, rL, omega1, gmat1, giL)
            + _dho_conv_pixel_integrated(x, amp2, rR, omega2, gmat2, giR)
            + offset
        )

    return _2dho_conv_anchored
