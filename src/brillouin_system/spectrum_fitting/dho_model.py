import numpy as np


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


def _2dho_binned(x, amp1, cen1, gamma1, amp2, cen2, gamma2, omega0, offset):
    """
    Joint two-peak DHO model with shared omega0 and independent gamma1/gamma2.

    cen1 and cen2 are the visible peak-maximum positions (cen1 < cen2). The
    elastic-line anchors are derived from them: the left peak is the Stokes
    hump of the Rayleigh order to its left (anchor at cen1 - u_pk1), the right
    peak is the anti-Stokes hump of the next order (anchor at cen2 + u_pk2).

    Each peak keeps its own damping gamma, so the two peaks may have different
    widths (e.g. from non-uniform spectrometer dispersion across the window).
    omega0 stays shared as a single, weakly-constrained asymmetry scale.

    Parameter order:
        [amp1, cen1, gamma1, amp2, cen2, gamma2, omega0, offset]
    """
    u_pk1 = dho_peak_offset(omega0, gamma1)
    u_pk2 = dho_peak_offset(omega0, gamma2)
    rayleigh_left = cen1 - u_pk1
    rayleigh_right = cen2 + u_pk2
    return (
        _dho_pixel_integrated(x, amp1, rayleigh_left, omega0, gamma1)
        + _dho_pixel_integrated(x, amp2, rayleigh_right, omega0, gamma2)
        + offset
    )


def _sort_2dho_params(popt):
    """Ensure the peak at the smaller pixel position comes first.

    popt = [amp1, cen1, gamma1, amp2, cen2, gamma2, omega0, offset]
    """
    popt = np.asarray(popt, dtype=float)
    if popt[4] < popt[1]:
        return np.array([
            popt[3], popt[4], popt[5],
            popt[0], popt[1], popt[2],
            popt[6], popt[7],
        ])
    return popt
