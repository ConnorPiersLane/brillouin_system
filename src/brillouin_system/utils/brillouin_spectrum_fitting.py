
from scipy.optimize import curve_fit

import numpy as np


from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.utils.fit_util import find_brillouin_peak_locations, select_top_two_peaks, _2Lorentzian, \
    sort_lorentzian_peaks, refine_fitted_spectrum


def get_fitted_spectrum_lorentzian(sline: np.ndarray, is_reference_mode: bool) -> FittedSpectrum:

    pix = np.arange(sline.shape[0])
    pk_ind, pk_info = find_brillouin_peak_locations(sline, is_reference_mode=is_reference_mode)


    if len(pk_ind) < 1:
        return FittedSpectrum(
            is_success=False,
            sline=sline,
            x_pixels=pix,
        )

    pk_ind, pk_info = select_top_two_peaks(pk_ind, pk_info)

    pk_wids = 0.5 * pk_info['widths']
    pk_hts = np.pi * pk_wids * pk_info['peak_heights']

    if len(pk_ind) == 1:
        offset = max(int(0.02 * len(sline)), 1)
        pk_ind = np.array([pk_ind[0] - offset, pk_ind[0] + offset])
        pk_wids = np.array([pk_wids[0], pk_wids[0]])
        pk_hts = np.array([pk_hts[0] / 2, pk_hts[0] / 2])

    p0 = [pk_hts[0], pk_ind[0], pk_wids[0],
          pk_hts[1], pk_ind[1], pk_wids[1], np.amin(sline)]

    try:
        popt, _ = curve_fit(_2Lorentzian, pix, sline, p0=p0)

        # Ensure peak 1 is left, peak 2 is right
        popt = sort_lorentzian_peaks(popt)

        interPeaksteps = np.abs(popt[4] - popt[1])
        fittedSpect = _2Lorentzian(pix, *popt)

        x_fit, y_fit = refine_fitted_spectrum(_2Lorentzian, pix, popt, factor=10)

        fitted_spectrum = FittedSpectrum(
            is_success=True,
            sline=sline,
            x_pixels=pix,
            fitted_spectrum=fittedSpect,
            x_fit_refined=x_fit,
            y_fit_refined=y_fit,
            lorentzian_parameters=popt,
            left_peak_center_px=float(popt[1]),
            left_peak_width_px=float(popt[2]),
            left_peak_amplitude=float(popt[0]),
            right_peak_center_px=float(popt[4]),
            right_peak_width_px=float(popt[5]),
            right_peak_amplitude=float(popt[3]),
            inter_peak_distance=interPeaksteps
        )

    except Exception as e:
        print(f"[fitSpectrum] Fitting failed: {e}")
        return FittedSpectrum(
            is_success=False,
            sline=sline,
            x_pixels=pix,
        )

    return fitted_spectrum

