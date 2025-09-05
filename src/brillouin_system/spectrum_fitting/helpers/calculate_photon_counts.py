from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum


@dataclass
class PhotonsCounts:
    left_peak_photons: float | None
    right_peak_photons: float | None
    total_photons: float | None


def calculate_photon_counts_from_fitted_spectrum(fs: FittedSpectrum,
                                                 preamp_gain: int | float,
                                                 emccd_gain: int | float) -> PhotonsCounts:

    if not fs.is_success:
        return PhotonsCounts(
            left_peak_photons=None,
            right_peak_photons=None,
            total_photons=None,
        )


    if emccd_gain == 0:
        count_to_electron_factor = preamp_gain
    else:
        count_to_electron_factor = preamp_gain / emccd_gain

    amp_l = fs.left_peak_amplitude
    amp_r = fs.right_peak_amplitude
    width_l = fs.left_peak_width_px
    width_r = fs.right_peak_width_px
    left_peak_photons = np.pi * amp_l * width_l * count_to_electron_factor
    right_peak_photons = np.pi * amp_r * width_r * count_to_electron_factor
    return PhotonsCounts(
        left_peak_photons=left_peak_photons,
        right_peak_photons=right_peak_photons,
        total_photons=left_peak_photons + right_peak_photons,
    )