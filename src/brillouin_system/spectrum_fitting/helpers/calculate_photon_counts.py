from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum

# Camera sensitivity at preamp 1x, i.e. photoelectrons per digitised count (ADU).
#
# MEASURED 2026-07-27 by photon transfer on the Andor DU897 (serial 9303) in
# Conventional amplifier mode, 0.08 MHz readout, preamp 1x: pixel temporal
# variance plotted against pixel mean over 50 repeated frames, variance taken
# from consecutive-frame differences so slow drift cannot inflate it.
# Slope-based estimate (independent of the bias level) 3.42, bright-pixel
# estimate 3.56, four water sessions -> 3.5 +- 0.5 e-/count.
# Cross-checks: background variance implies 4.3 e- read noise (datasheet 4-6 e-;
# a sensitivity of 1.0 would imply an impossible 1.2 e-), and the shot-noise
# bound computed with this value reproduces the measured fitted-shift scatter
# across five liquid datasets at a ratio of 1.00 +- 0.07.
#
# NOTE: the camera API's "preamp gain" is the preamp MULTIPLIER (1x, 2x, 3x),
# not the sensitivity, despite older docstrings calling it e-/count. Raising the
# multiplier lowers the electrons per count, hence it divides below.
SENSITIVITY_E_PER_COUNT_PREAMP_1X = 3.5


@dataclass
class PhotonsCounts:
    left_peak_photons: float | None
    right_peak_photons: float | None
    total_photons: float | None


def electrons_per_count(preamp_gain: int | float,
                        emccd_gain: int | float,
                        sensitivity_e_per_count: float = SENSITIVITY_E_PER_COUNT_PREAMP_1X,
                        ) -> float:
    """
    Photoelectrons represented by one digitised count.

    preamp_gain: preamp multiplier (1x, 2x, 3x) as reported by the camera.
    emccd_gain:  EM gain, or 0 when the EM register is not in use.

    Electrons referred to the sensor are counts * sensitivity / (preamp * em).

    Caveat for EM mode: the EM register also adds a stochastic multiplication
    noise (excess noise factor ~sqrt(2)), so the shot-noise limit is worse than
    the electron count alone suggests. That factor is not applied here; the
    system currently runs in Conventional mode where it does not arise.
    """
    factor = sensitivity_e_per_count / max(preamp_gain, 1e-12)
    if emccd_gain:
        factor /= emccd_gain
    return factor


def count_to_electrons(counts: int | float,
                       preamp_gain: int | float,
                       emccd_gain: int | float,
                       sensitivity_e_per_count: float = SENSITIVITY_E_PER_COUNT_PREAMP_1X,
                       ) -> float:
    return electrons_per_count(preamp_gain, emccd_gain, sensitivity_e_per_count) * counts


def calculate_photon_counts_from_fitted_spectrum(fs: FittedSpectrum,
                                                 preamp_gain: int | float,
                                                 emccd_gain: int | float,
                                                 sensitivity_e_per_count: float = SENSITIVITY_E_PER_COUNT_PREAMP_1X,
                                                 ) -> PhotonsCounts:
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
        sensitivity_e_per_count=sensitivity_e_per_count,
    )
    right_peak_photons = count_to_electrons(
        counts=np.pi * fs.right_peak_amplitude * fs.right_peak_width_px,
        preamp_gain=preamp_gain,
        emccd_gain=emccd_gain,
        sensitivity_e_per_count=sensitivity_e_per_count,
    )

    return PhotonsCounts(
        left_peak_photons=left_peak_photons,
        right_peak_photons=right_peak_photons,
        total_photons=left_peak_photons + right_peak_photons,
    )
