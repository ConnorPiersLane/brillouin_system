import numpy as np
from scipy.special import wofz

from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.utils.brillouin_spectrum_fitting import get_fitted_spectrum_generic

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

def _2Voigt(x, amp1, cen1, gamma1, amp2, cen2, gamma2, offset, sigma_func_left, sigma_func_right):
    """
    Double Voigt model with pixel-dependent Gaussian sigma (PSF) via separate left/right functions.
    """
    sigma1 = sigma_func_left(cen1)
    sigma2 = sigma_func_right(cen2)

    profile1 = _voigt_profile(x, amp1, cen1, gamma1, sigma1)
    profile2 = _voigt_profile(x, amp2, cen2, gamma2, sigma2)

    return profile1 + profile2 + offset

def _generate_initial_guess_voigt(pk_ind, pk_info, sline):
    """
    Generate initial guess parameters for a double Voigt fit.
    """
    pk_wids = 0.5 * pk_info['widths']
    pk_hts = pk_info['peak_heights']

    if len(pk_ind) == 1:
        offset = max(int(0.02 * len(sline)), 1)
        pk_ind = np.array([pk_ind[0] - offset, pk_ind[0] + offset])
        pk_wids = np.array([pk_wids[0], pk_wids[0]])
        pk_hts = np.array([pk_hts[0] / 2, pk_hts[0] / 2])

    p0 = [
        pk_hts[0], pk_ind[0], pk_wids[0],
        pk_hts[1], pk_ind[1], pk_wids[1],
        np.amin(sline)
    ]
    return p0

def voigt_model_wrapper(sigma_func_left, sigma_func_right):
    """
    Returns a function compatible with curve_fit to evaluate the double Voigt profile.
    """
    def model(x, amp1, cen1, gamma1, amp2, cen2, gamma2, offset):
        return _2Voigt(x, amp1, cen1, gamma1, amp2, cen2, gamma2, offset,
                       sigma_func_left, sigma_func_right)
    return model

def get_fitted_spectrum_voigt(sline: np.ndarray,
                              is_reference_mode: bool,
                              sigma_func_left,
                              sigma_func_right) -> FittedSpectrum:
    """
    Fit Brillouin spectrum using double Voigt profiles, with separate PSF sigma functions.
    """
    return get_fitted_spectrum_generic(
        sline,
        is_reference_mode,
        model_function=voigt_model_wrapper(sigma_func_left, sigma_func_right),
        p0_generator=_generate_initial_guess_voigt
    )
