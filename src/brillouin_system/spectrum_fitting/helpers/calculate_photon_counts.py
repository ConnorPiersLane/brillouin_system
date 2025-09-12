from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum


@dataclass
class PhotonsCounts:
    left_peak_photons: float | None
    right_peak_photons: float | None
    total_photons: float | None


def count_to_electrons(counts: int | float,
                       preamp_gain: int | float,
                       emccd_gain: int | float) -> float:
    if emccd_gain == 0:
        count_to_electron_factor = preamp_gain
    else:
        count_to_electron_factor = preamp_gain / emccd_gain
    return count_to_electron_factor * counts


def calculate_photon_counts_from_fitted_spectrum(fs: FittedSpectrum,
                                                 preamp_gain: int | float,
                                                 emccd_gain: int | float) -> PhotonsCounts:
    if not fs.is_success:
        return PhotonsCounts(
            left_peak_photons=None,
            right_peak_photons=None,
            total_photons=None,
        )

    left_peak_photons = count_to_electrons(
        counts=np.pi * fs.left_peak_amplitude * fs.left_peak_width_px,
        preamp_gain=preamp_gain,
        emccd_gain=emccd_gain,
    )
    right_peak_photons = count_to_electrons(
        counts=np.pi * fs.right_peak_amplitude * fs.right_peak_width_px,
        preamp_gain=preamp_gain,
        emccd_gain=emccd_gain,
    )

    return PhotonsCounts(
        left_peak_photons=left_peak_photons,
        right_peak_photons=right_peak_photons,
        total_photons=left_peak_photons + right_peak_photons,
    )
