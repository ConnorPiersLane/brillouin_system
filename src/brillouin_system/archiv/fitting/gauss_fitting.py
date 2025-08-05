import numpy as np

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.fit_util import get_fitted_spectrum_generic


def _2Gaussian(x, amp1, cen1, wid1, amp2, cen2, wid2, offs):
    """
    Two Gaussian peaks plus an offset.
    amp1, cen1, wid1: amplitude, center, and standard deviation of peak 1
    amp2, cen2, wid2: amplitude, center, and standard deviation of peak 2
    offs: constant offset
    """
    return (amp1 * np.exp(-0.5 * ((x - cen1) / wid1) ** 2)) + \
           (amp2 * np.exp(-0.5 * ((x - cen2) / wid2) ** 2)) + offs

def _generate_initial_guess_gaussian(pk_ind, pk_info, sline):
    """
    Generate initial guess parameters for a double Gaussian fit.
    """
    pk_wids = 0.5 * pk_info['widths']
    pk_hts = pk_info['peak_heights']  # Use raw heights for Gaussian

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


def get_fitted_spectrum_gaussian(sline: np.ndarray, is_reference_mode: bool) -> FittedSpectrum:
    """
    Fit using a double Gaussian model.
    """
    return get_fitted_spectrum_generic(
        sline,
        is_reference_mode,
        model_function=_2Gaussian,
        p0_generator=_generate_initial_guess_gaussian
    )
