import numpy as np

from brillouin_system.fitting.fit_util import get_fitted_spectrum_generic
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum




def _2Lorentzian(x, amp1, cen1, wid1, amp2, cen2, wid2, offs):
    return (amp1 * wid1 ** 2 / ((x - cen1) ** 2 + wid1 ** 2)) + \
           (amp2 * wid2 ** 2 / ((x - cen2) ** 2 + wid2 ** 2)) + offs

def _generate_initial_guess_lorentzian(pk_ind, pk_info, sline):
    """
    Generate initial guess parameters for a double Lorentzian fit.
    """
    pk_wids = 0.5 * pk_info['widths']
    pk_hts = np.pi * pk_wids * pk_info['peak_heights']  # Lorentzian area estimate

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






def get_fitted_spectrum_lorentzian(sline: np.ndarray, is_reference_mode: bool) -> FittedSpectrum:
    """
    Fit using a double Lorentzian model.
    """
    return get_fitted_spectrum_generic(
        sline,
        is_reference_mode,
        model_function=_2Lorentzian,
        p0_generator=_generate_initial_guess_lorentzian
    )

