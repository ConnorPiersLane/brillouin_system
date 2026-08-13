"""
Photon (photoelectron) counts behind a fitted spectrum.

Everything here exists to answer one question: how many photoelectrons produced
this peak? That number feeds the shot-noise precision in spectrum_analyzer, so
an error in it moves every theoretical uncertainty we quote.


HOW THE SENSITIVITY WAS MEASURED (2026-08-12, superseding 2026-07-27)
---------------------------------------------------------------------
Light arrives in Poisson-distributed lumps, so a pixel holding N electrons has
variance N. In counts alone that would give

    variance[ADU^2] = signal[ADU] / g          (g = electrons per count)

but on real spectra a second term is NOT negligible: the source carries a
per-frame common-mode intensity fluctuation (laser power / fiber coupling,
measured independently at ~1% rms by the Stokes/anti-Stokes covariance
analysis), which every pixel sees multiplicatively. The honest model is

    variance[ADU^2] = S/g + (eps*S)^2 + const

Fitted to 126 scans x 50 frames of the 2026-8-3 / 2026-8-5 water temperature
series (camera_gain_temperature_series.py in Data/Calibration_paper_data):

    g   = 3.89 +- 0.04 e-/count   (per day: 3.91 / 3.90 -- hardware-stable)
    eps = 1.5% / 1.2% per day     (session-dependent, matches the covariance
                                   study's common mode)

A LINEAR photon-transfer fit lumps the quadratic term into the slope and
returns a gain biased LOW by a session-dependent amount: 2.9 on this series,
3.5 on the July sessions (the value this constant held until 2026-08-12), and
that bias is also why July's per-session estimates spanned 2.7-4.8. The
common-mode term scales the whole spectrum without moving peak centres, so
shot-noise predictions for the fitted SHIFT must use the Poisson gain g.

Cross-checks on 3.89: implied read noise 1.26 ADU * 3.89 = 4.9 e- (DU897
datasheet 4-6); Andor quotes 4-5 e-/count for this camera in Conventional
mode; and with 3.89 the full noise budget of the fitted shift closes with no
free parameters (see theoretical_precision's docstring).

TWO TRAPS in photon-transfer measurements, both of which we fell into first:

  * Slow drift. Over 50 frames the source drifts, inflating a naive temporal
    variance by ~22% at bright pixels. Take the variance from CONSECUTIVE-FRAME
    DIFFERENCES instead; anything slower than the frame rate cancels. (The
    ~1% common mode above is frame-to-frame, so differencing does NOT remove
    it -- hence the quadratic term.)

  * Outlier rejection on few frames. With k frames the per-pixel variance has
    k-1 degrees of freedom and a long right tail. Rejecting "outliers" cuts that
    tail and biases the gain HIGH. Measured cost: the same water data cut to
    3-frame chunks returned 4.88 e-/count with rejection versus 3.32 without.
    Use >=20 frames, and do not reject.

The MATH here is verified: the area of the pixel-integrated Lorentzian is
exactly pi*amp*width (the telescoping arctan sum), and the preamp multiplier
divides, matching both the physics and Andor's own convention that a higher
preamp setting yields a LOWER e-/count.

The definitive gain measurement has still NOT been done: a flat, uniform
illumination at an exposure series would remove the common-mode/Poisson
separation from a model fit entirely. ~15 minutes at the instrument.

The EM branch is derived but UNTESTED -- we own no EM-mode data. Its
sensitivity is unmeasured and the code raises rather than guess.
"""
from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum

# Photoelectrons per digitised count (ADU) at preamp 1x, for the ONE amplifier
# mode we have calibrated: Conventional, 0.08 MHz readout, on the Andor DU897
# (serial 9303). Sensitivity is a function of (output amplifier, readout speed,
# preamp) -- it is NOT a single number for the camera, so this does not
# generalise to another mode. Measured 2026-08-12 by quadratic photon transfer
# (var = S/g + (eps*S)^2 + c); a linear fit is biased low by the common-mode
# source noise and gave the 3.5 used before. See the module docstring.
#
# The camera API's "preamp gain" is the preamp MULTIPLIER (1x, 2x, 3x), not the
# sensitivity, despite older docstrings calling it e-/count. A higher multiplier
# means FEWER electrons per count, hence it divides below.
SENSITIVITY_E_PER_COUNT_PREAMP_1X = 3.89

# Electron-Multiplying mode runs through a different output amplifier, so the
# Conventional sensitivity above does NOT carry over. Measure it the same way
# (photon transfer, >=20 frames, NO outlier rejection -- see the module
# docstring; rejection on few frames biases the gain high by 60-80%) and put the
# number here. Until then the EM path raises rather than guess.
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

    # Peak area in counts is exactly pi * amp * width: summing the
    # pixel-integrated Lorentzian amp*w*[arctan((x+.5-c)/w) - arctan((x-.5-c)/w)]
    # over all pixels telescopes to amp*w*[arctan(inf) - arctan(-inf)] = amp*w*pi.
    # Exact for any width, so narrow peaks need no correction.
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
