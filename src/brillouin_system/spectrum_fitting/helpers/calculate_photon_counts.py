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
#
# THIS VALUE IS SPECIFIC TO ONE AMPLIFIER MODE. Sensitivity depends on the
# output amplifier (Conventional vs Electron Multiplying) and on the readout
# speed; it is not a single number for the camera. Switching modes requires
# re-measuring it.
SENSITIVITY_E_PER_COUNT_PREAMP_1X = 3.5

# Electron-Multiplying mode runs through a different output amplifier, so the
# Conventional sensitivity above does NOT carry over. Measure it the same way
# (photon transfer, >=20 frames, no outlier rejection -- rejection on few frames
# biases the gain high by 60-80%) and put the number here.
SENSITIVITY_E_PER_COUNT_EM_PREAMP_1X: float | None = None

# The EM register multiplies stochastically, which doubles the variance of the
# photon term (excess noise factor F^2 = 2, i.e. sqrt(2) on sigma). Equivalent
# to using N/2 photons in a shot-noise bound. Not applied by
# theoretical_precision -- see the note in electrons_per_count.
EM_EXCESS_NOISE_FACTOR = 2.0


@dataclass
class PhotonsCounts:
    left_peak_photons: float | None
    right_peak_photons: float | None
    total_photons: float | None


def electrons_per_count(preamp_gain: int | float,
                        emccd_gain: int | float,
                        sensitivity_e_per_count: float | None = None,
                        ) -> float:
    """
    Photoelectrons at the sensor represented by one digitised count.

    preamp_gain: preamp multiplier (1x, 2x, 3x) as reported by the camera.
    emccd_gain:  EM gain, or 0 when the EM register is not in use.
    sensitivity_e_per_count: overrides the mode-derived constant.

    counts * sensitivity / (preamp * em).

    EM mode needs SENSITIVITY_E_PER_COUNT_EM_PREAMP_1X to be measured first --
    the Conventional value does not transfer, because EM mode reads out through
    a different amplifier. Rather than return a wrong number this raises.

    NOTE for EM mode: the electron count returned here is the honest number of
    photoelectrons, but a shot-noise bound built from it will be optimistic by
    sqrt(EM_EXCESS_NOISE_FACTOR), since the EM register multiplies
    stochastically. Feed N / EM_EXCESS_NOISE_FACTOR to a precision calculation,
    or scale its result, when running in EM mode.
    """
    if preamp_gain is None or preamp_gain <= 0:
        raise ValueError(f"preamp multiplier must be positive, got {preamp_gain!r}")

    if sensitivity_e_per_count is None:
        if emccd_gain:
            sensitivity_e_per_count = SENSITIVITY_E_PER_COUNT_EM_PREAMP_1X
            if sensitivity_e_per_count is None:
                raise ValueError(
                    "Camera is in Electron-Multiplying mode but the EM sensitivity "
                    "has never been measured. The Conventional-mode value "
                    f"({SENSITIVITY_E_PER_COUNT_PREAMP_1X} e-/count) does not apply: EM "
                    "mode uses a different output amplifier. Measure it by photon "
                    "transfer and set SENSITIVITY_E_PER_COUNT_EM_PREAMP_1X, or pass "
                    "sensitivity_e_per_count explicitly. Remember EM also carries an "
                    f"excess noise factor of {EM_EXCESS_NOISE_FACTOR} on the variance."
                )
        else:
            sensitivity_e_per_count = SENSITIVITY_E_PER_COUNT_PREAMP_1X

    factor = sensitivity_e_per_count / preamp_gain
    if emccd_gain:
        factor /= emccd_gain
    return factor


def count_to_electrons(counts: int | float,
                       preamp_gain: int | float,
                       emccd_gain: int | float,
                       sensitivity_e_per_count: float | None = None,
                       ) -> float:
    return electrons_per_count(preamp_gain, emccd_gain, sensitivity_e_per_count) * counts


def calculate_photon_counts_from_fitted_spectrum(fs: FittedSpectrum,
                                                 preamp_gain: int | float,
                                                 emccd_gain: int | float,
                                                 sensitivity_e_per_count: float | None = None,
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
