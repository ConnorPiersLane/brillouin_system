import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz

from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.utils.fit_util import (
    find_peak_locations,
    select_top_two_peaks,
    sort_peaks,
    refine_fitted_spectrum
)


def voigt_fwhm(gamma, sigma):
    gamma_fwhm = 2 * gamma
    sigma_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    fwhm_voigt = 0.5346 * gamma_fwhm + np.sqrt(0.2166 * gamma_fwhm**2 + sigma_fwhm**2)
    return fwhm_voigt

def voigt_sigma(gamma, sigma):
    fwhm_v = voigt_fwhm(gamma, sigma)
    return fwhm_v / (2 * np.sqrt(2 * np.log(2)))


def _voigt_profile(x, amp, cen, gamma, sigma):
    """
    Single Voigt profile evaluated at x, given amplitude, center, Lorentzian HWHM (gamma),
    and Gaussian sigma.
    """
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    voigt = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return amp * voigt


def _2Voigt(x, amp1, cen1, gamma1, amp2, cen2, gamma2, offset, sigma_func):
    """
    Double Voigt model with pixel-dependent Gaussian sigma (PSF) via function.
    """
    sigma1 = sigma_func(cen1)
    sigma2 = sigma_func(cen2)

    profile1 = _voigt_profile(x, amp1, cen1, gamma1, sigma1)
    profile2 = _voigt_profile(x, amp2, cen2, gamma2, sigma2)

    return profile1 + profile2 + offset


def _generate_initial_guess_voigt(pk_ind, pk_info, sline):
    """
    Generate initial guess parameters for a double Voigt fit.
    """
    pk_wids = 0.5 * pk_info['widths']
    pk_hts = pk_info['peak_heights']  # use heights initially

    if len(pk_ind) == 1:
        offset = max(int(0.02 * len(sline)), 1)
        pk_ind = np.array([pk_ind[0] - offset, pk_ind[0] + offset])
        pk_wids = np.array([pk_wids[0], pk_wids[0]])
        pk_hts = np.array([pk_hts[0] / 2, pk_hts[0] / 2])

    # initial guess: amp, center, Lorentzian width
    p0 = [
        pk_hts[0], pk_ind[0], pk_wids[0],  # peak 1
        pk_hts[1], pk_ind[1], pk_wids[1],  # peak 2
        np.amin(sline)                    # offset
    ]
    return p0


def get_fitted_spectrum_voigt(sline: np.ndarray,
                              is_reference_mode: bool,
                              sigma_func) -> FittedSpectrum:
    """
    Fit Brillouin spectrum using double Voigt profiles,
    with PSF sigma from a user-provided function sigma_func(center_px).
    """
    pix = np.arange(sline.shape[0])
    sline = np.clip(sline, 0, None)

    pk_ind, pk_info = find_peak_locations(sline, is_reference_mode=is_reference_mode)
    if len(pk_ind) < 1:
        return FittedSpectrum(
            is_success=False,
            sline=sline,
            x_pixels=pix,
        )

    pk_ind, pk_info = select_top_two_peaks(pk_ind, pk_info)
    p0 = _generate_initial_guess_voigt(pk_ind, pk_info, sline)

    try:
        n_pix = len(pix)
        lower_bounds = [0, 0, 0, 0, 0, 0, 0]  # amps, centers, widths >= 0
        upper_bounds = [np.inf, n_pix, n_pix/2, np.inf, n_pix, n_pix/2, np.inf]

        # Use a lambda to pass sigma_func as a fixed parameter
        def model(x, amp1, cen1, gamma1, amp2, cen2, gamma2, offset):
            return _2Voigt(x, amp1, cen1, gamma1, amp2, cen2, gamma2, offset, sigma_func)

        popt, _ = curve_fit(
            model,
            pix,
            sline,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )

        amp1, cen1, gamma1, amp2, cen2, gamma2, offset = sort_peaks(popt)

        fittedSpect = model(pix, amp1, cen1, gamma1, amp2, cen2, gamma2, offset)
        x_fit, y_fit = refine_fitted_spectrum(model, pix, popt, factor=10)

        fitted_spectrum = FittedSpectrum(
            is_success=True,
            sline=sline,
            x_pixels=pix,
            fitted_spectrum=fittedSpect,
            x_fit_refined=x_fit,
            y_fit_refined=y_fit,
            parameters=popt,
            left_peak_center_px=float(cen1),
            left_peak_width_px=float(gamma1),
            left_peak_amplitude=float(amp1),
            right_peak_center_px=float(cen2),
            right_peak_width_px=float(gamma2),
            right_peak_amplitude=float(amp2),
            inter_peak_distance=np.abs(cen2 - cen1)
        )

    except Exception as e:
        print(f"[fitSpectrum] Voigt fitting failed: {e}")
        return FittedSpectrum(
            is_success=False,
            sline=sline,
            x_pixels=pix,
        )

    return fitted_spectrum
