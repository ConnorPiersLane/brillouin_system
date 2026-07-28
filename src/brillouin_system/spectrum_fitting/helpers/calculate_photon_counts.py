"""
Photon (photoelectron) counts behind a fitted spectrum.

Everything here exists to answer one question: how many photoelectrons produced
this peak? That number feeds the shot-noise precision in spectrum_analyzer, so
an error in it moves every theoretical uncertainty we quote.


HOW THE SENSITIVITY WAS MEASURED (2026-07-27)
---------------------------------------------
Light arrives in Poisson-distributed lumps, so a pixel holding N electrons has
variance N. In counts that becomes

    variance[ADU^2] = signal[ADU] / g          (g = electrons per count)

so plotting each pixel's temporal variance against its mean gives a line of
slope 1/g. Applied to 50 repeated frames of the same water spectrum, over 1760
pixels.

Two independent estimators, four water sessions:
    slope of variance vs mean, all pixels     3.48 +- 0.63 e-/count
    (mean - bias)/variance, brightest 1%      3.56 +- 0.56 e-/count

TWO TRAPS, both of which we fell into first:

  * Slow drift. Over 50 frames the source drifts, inflating a naive temporal
    variance by ~22% at bright pixels. Take the variance from CONSECUTIVE-FRAME
    DIFFERENCES instead; anything slower than the frame rate cancels.

  * Outlier rejection on few frames. With k frames the per-pixel variance has
    k-1 degrees of freedom and a long right tail. Rejecting "outliers" cuts that
    tail and biases the gain HIGH. Measured cost: the same water data cut to
    3-frame chunks returned 4.88 e-/count with rejection versus 3.32 without,
    and cornea 3-frame data returned 5.8-6.7 with rejection versus 3.2-4.1
    without. Use >=20 frames, and do not reject.


WHY WE BELIEVE 3.5 RATHER THAN THE 1.0 THIS CODE USED TO ASSUME
---------------------------------------------------------------
  * Read noise. Background pixels carry 1.49 ADU^2, i.e. 1.22 ADU. At 1.0
    e-/count that is a read noise of 1.2 e-, which no CCD achieves without an
    EM register; at 3.5 it is 4.3 e-, matching the DU897 datasheet's 4-6 e-.
    This check involves no fitting at all.
  * Prediction. The sensitivity was measured from pixel noise, then used to
    predict a different quantity -- the scatter of the fitted Brillouin shift --
    across five liquid datasets. Measured/predicted came out 1.00 +- 0.07.
    Inverting that gives an independent 3.5 (range 3.0-4.0). At 1.0 the
    prediction would have missed by 47% on all five.


HOW FAR TO TRUST IT
-------------------
The MATH here is verified: the area of the pixel-integrated Lorentzian is
exactly pi*amp*width (the telescoping arctan sum), and the preamp multiplier
divides, matching both the physics and Andor's own convention that a higher
preamp setting yields a LOWER e-/count.

The VALUE carries real uncertainty. Per-session estimates spanned 2.7-4.8.
Gain is fixed hardware, so that spread is measurement systematic, not drift --
call it +-15%, which propagates to about +-8% on any sigma (sigma ~ 1/sqrt(g)).
Note also that Andor typically quotes 4-5 e-/count for this camera in
Conventional mode, so 3.5 may sit ~20% low.

The definitive measurement has NOT been done: a flat, uniform illumination at a
series of exposure times, sweeping the full dynamic range instead of relying on
whatever range a spectrum happens to span. That is a ~15 minute experiment and
would take +-15% down to a few percent.

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
# generalise to another mode. See the module docstring for the measurement,
# its pitfalls, and how far to trust the value.
#
# The camera API's "preamp gain" is the preamp MULTIPLIER (1x, 2x, 3x), not the
# sensitivity, despite older docstrings calling it e-/count. A higher multiplier
# means FEWER electrons per count, hence it divides below.
SENSITIVITY_E_PER_COUNT_PREAMP_1X = 3.5

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
