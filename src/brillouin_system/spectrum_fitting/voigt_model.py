import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile


def _voigt_pixel_integrated(x, amp, cen, gamma, sigma_psf, oversample=25):
    """
    Pixel-integrated Voigt model.

    gamma     = Lorentzian half-width parameter
    sigma_psf = Gaussian blur / effective instrument response width
    amp       = area-like amplitude scale
    cen       = fitted center in pixel units
    """
    x = np.asarray(x, dtype=float)

    u = np.linspace(-0.5, 0.5, oversample)
    xx = x[:, None] + u[None, :]

    y = voigt_profile(xx - cen, sigma_psf, gamma)

    return amp * np.trapz(y, u, axis=1)


def _1voigt_binned(x, amp, cen, gamma, sigma_psf, offset):
    return _voigt_pixel_integrated(x, amp, cen, gamma, sigma_psf) + offset


def _1voigt_binned_slope(x, amp, cen, gamma, sigma_psf, offset, slope):
    x0 = np.mean(x)
    return (
        _voigt_pixel_integrated(x, amp, cen, gamma, sigma_psf)
        + offset
        + slope * (x - x0)
    )


def _2voigt_binned(
    x,
    amp1, cen1, gamma1, sigma_psf1,
    amp2, cen2, gamma2, sigma_psf2,
    offset,
):
    """
    Two Voigt peaks with independent sigma_psf values.

    Parameter order:
        [amp1, cen1, gamma1, sigma_psf1,
         amp2, cen2, gamma2, sigma_psf2,
         offset]
    """
    return (
        _voigt_pixel_integrated(x, amp1, cen1, gamma1, sigma_psf1)
        + _voigt_pixel_integrated(x, amp2, cen2, gamma2, sigma_psf2)
        + offset
    )



def fit_one_voigt_binned(px, sline, cen0, gamma0=0.7):
    px = np.asarray(px, dtype=float)
    sline = np.asarray(sline, dtype=float)

    offset0 = float(np.min(sline))
    amp0 = float(np.sum(sline - offset0))
    sigma_psf0 = 0.25

    p0 = [amp0, cen0, gamma0, sigma_psf0, offset0]

    bounds = (
        [0.0, np.min(px), 0.03, 0.00, -np.inf],
        [np.inf, np.max(px), 10.0, 5.00, np.inf],
    )

    popt, pcov = curve_fit(
        _1voigt_binned,
        px,
        sline,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )

    amp, cen, gamma, sigma_psf, offset = popt

    return {
        "popt": popt,
        "pcov": pcov,
        "center_px": float(cen),
        "gamma_px": float(gamma),
        "sigma_psf_px": float(sigma_psf),
        "offset": float(offset),
    }


