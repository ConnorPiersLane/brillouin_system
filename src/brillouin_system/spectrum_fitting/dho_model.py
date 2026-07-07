from dataclasses import dataclass

import numpy as np


@dataclass
class ElasticAnchors:
    """
    Pixel positions of the elastic (Rayleigh) lines bracketing the two
    Brillouin peaks, derived from the frequency calibration (see
    CalibrationCalculator.elastic_anchors). The left Rayleigh order sits left
    of the left peak, the right order right of the right peak.
    """
    rayleigh_left_px: float
    rayleigh_right_px: float


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


def dho_peak_offset(omega0, gamma):
    """Offset of the intensity maximum from the elastic line: sqrt(omega0^2 - gamma^2/2)."""
    return float(np.sqrt(max(omega0**2 - 0.5 * gamma**2, 1e-12)))


def dho_peak_height(omega0, gamma):
    """Value of dho_intensity at its maximum (for unit amplitude)."""
    denom = gamma * max(omega0**2 - 0.25 * gamma**2, 1e-12)
    return omega0**2 / denom


def _dho_pixel_integrated(x, amp, rayleigh_px, omega0, gamma, oversample=100):
    """
    Pixel-integrated DHO, anchored at the elastic-line position rayleigh_px.

    amp         = amplitude scale factor (peak height = amp * dho_peak_height)
    rayleigh_px = pixel position of the elastic (Rayleigh) line
    """
    x = np.asarray(x, dtype=float)

    u = np.linspace(-0.5, 0.5, oversample)
    xx = x[:, None] + u[None, :]

    y = dho_intensity(xx - rayleigh_px, omega0, gamma)

    return amp * np.trapezoid(y, u, axis=1)


def make_2dho_anchored(rayleigh_left_px, rayleigh_right_px):
    """
    Build a two-peak DHO model (eq. S2 twice) with fixed elastic-line anchors.

    Each peak is a DHO anchored at its own Rayleigh order: the left peak is
    the Stokes hump of the left order (visible at rayleigh_left_px + u_pk1),
    the right peak the anti-Stokes hump of the right order (visible at
    rayleigh_right_px - u_pk2). With the anchors fixed by the calibration,
    omega1 and omega2 are directly the Brillouin resonances of the two peaks
    in pixel units, measured from their own elastic lines. The fitted gammas
    are total (material + instrument) widths; instrument subtraction happens
    downstream.

    Parameter order:
        [amp1, omega1, gamma1, amp2, omega2, gamma2, offset]
    """
    rayleigh_left_px = float(rayleigh_left_px)
    rayleigh_right_px = float(rayleigh_right_px)

    def _2dho_anchored(x, amp1, omega1, gamma1, amp2, omega2, gamma2, offset):
        return (
            _dho_pixel_integrated(x, amp1, rayleigh_left_px, omega1, gamma1)
            + _dho_pixel_integrated(x, amp2, rayleigh_right_px, omega2, gamma2)
            + offset
        )

    return _2dho_anchored
