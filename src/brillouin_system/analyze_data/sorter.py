import numpy as np

from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum


def sort_fitted_spectrum_peaks(spectrum: FittedSpectrum) -> FittedSpectrum:
    """
    Sort the peaks in a FittedSpectrum so that the left peak comes before the right
    based on center pixel positions. This updates all relevant attributes.

    Parameters:
        spectrum (FittedSpectrum): The spectrum to correct.

    Returns:
        FittedSpectrum: The corrected spectrum with ordered peaks.
    """
    if not spectrum.is_success or spectrum.parameters is None:
        return spectrum  # No sorting needed or possible

    # Unpack parameters
    amp1, cen1, wid1, amp2, cen2, wid2, offset = spectrum.parameters

    if cen1 <= cen2:
        return spectrum  # Already ordered correctly

    # Swap peak attributes
    spectrum.left_peak_center_px, spectrum.right_peak_center_px = cen2, cen1
    spectrum.left_peak_width_px, spectrum.right_peak_width_px = wid2, wid1
    spectrum.left_peak_amplitude, spectrum.right_peak_amplitude = amp2, amp1
    spectrum.inter_peak_distance = abs(cen2 - cen1)

    # Sort lorentzian_parameters to reflect this
    spectrum.parameters = np.array([amp2, cen2, wid2, amp1, cen1, wid1, offset])

    return spectrum
