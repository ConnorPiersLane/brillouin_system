"""
Per-peak pixel counts and photon (photoelectron) numbers behind a fitted
spectrum.

Everything here exists to answer one question: how many counts and how many
photoelectrons produced each peak? The photon numbers feed the shot-noise
bound in noise_analysis.thompson, so an error here moves every theoretical
uncertainty we quote. Inputs are the FIT and the camera gain settings — never
the frame: summing the frame would fold in background, stray background and the
neighbour peak's tails, exactly the contaminations the fit decomposes away.
The peak area in counts is exact from the fit parameters (see from_fit).


THE NUMBERS LIVE IN ccd_characteristics (2026-08-20 rule: every parameter
that has to be fitted or obtained is written in ccd_characteristics.toml,
next to the measurement_scripts/ that produced it). This module reads them
from there at call time. In brief, so the code makes sense on its own:

  * sensitivity g [e-/count] is per (amplifier, readout speed, preamp) mode
    and was measured by QUADRATIC photon transfer, var = S/g + (eps*S)^2 + c
    -- the source's ~1% common-mode intensity noise makes a LINEAR PTC
    biased low (it gave 2.9-3.5). The common mode scales the spectrum
    without moving peak centres, so shift bounds must use the Poisson g.
    Full methodology + the two traps (drift -> consecutive-frame
    differences; outlier rejection biases high) in
    ccd_characteristics/measurement_scripts/measure_gain_photon_transfer.py.
  * The camera API's "preamp gain" is the preamp MULTIPLIER (1x, 2x, 3x),
    not the sensitivity. A higher multiplier means FEWER electrons per
    count, hence it divides in electrons_per_count.
  * The EM amplifier has its own (unmeasured) sensitivity -- the
    Conventional value does not transfer, so the EM path raises rather
    than guess. The EM register also multiplies stochastically (excess
    noise factor F^2 = 2 on the variance, i.e. N/2 effective photons in a
    shot-noise bound) -- not applied by theoretical_precision, see the
    note in electrons_per_count.

The MATH here is verified: the area of the pixel-integrated Lorentzian is
exactly pi*amp*width (the telescoping arctan sum), and the preamp multiplier
divides, matching both the physics and Andor's own convention that a higher
preamp setting yields a LOWER e-/count.
"""
from dataclasses import dataclass

import numpy as np

from brillouin_system.ccd_characteristics import ccd_config
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum

# Backwards-compatible module aliases, frozen at import time. The functions
# below read the LIVE ccd_config instead — prefer that in new code.
SENSITIVITY_E_PER_COUNT_PREAMP_1X = (
    ccd_config.get().sensitivity_e_per_count_preamp_1x)
SENSITIVITY_E_PER_COUNT_EM_PREAMP_1X: float | None = (
    ccd_config.get().sensitivity_e_per_count_em_preamp_1x or None)
EM_EXCESS_NOISE_FACTOR = ccd_config.get().em_excess_noise_factor


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
        ccd = ccd_config.get()
        if emccd_gain:
            sensitivity_e_per_count = ccd.sensitivity_e_per_count_em_preamp_1x
            if not sensitivity_e_per_count:
                raise ValueError(
                    "Camera is in Electron-Multiplying mode but the EM sensitivity "
                    "has never been measured. The Conventional-mode value "
                    f"({ccd.sensitivity_e_per_count_preamp_1x} e-/count) does not "
                    "apply: EM mode uses a different output amplifier. Measure it "
                    "by photon transfer (ccd_characteristics/measurement_scripts/"
                    "measure_gain_photon_transfer.py) and set "
                    "sensitivity_e_per_count_em_preamp_1x in "
                    "ccd_characteristics.toml, or pass sensitivity_e_per_count "
                    "explicitly. Remember EM also carries an excess noise factor "
                    f"of {ccd.em_excess_noise_factor} on the variance."
                )
        else:
            sensitivity_e_per_count = ccd.sensitivity_e_per_count_preamp_1x

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


@dataclass
class PixelCountsAndPhotons:
    """Per-peak signal behind a fit: counts (ADU) and photons (photoelectrons).

    The production chain is two peaks: with a 4-peak fit the reported
    left/right peaks are the inner main pair, so this stays pair-based.
    """
    left_peak_counts: float | None
    right_peak_counts: float | None
    left_peak_photons: float | None
    right_peak_photons: float | None
    total_counts: float | None
    total_photons: float | None

    @classmethod
    def from_fit(cls, fs: FittedSpectrum,
                 preamp_gain: int | float,
                 emccd_gain: int | float,
                 sensitivity_e_per_count: float | None = None,
                 ) -> "PixelCountsAndPhotons":
        """Counts and photons of each peak from the fit parameters alone.

        BACKGROUND-FREE BY CONSTRUCTION: the fit model is peaks + background,
        so amplitude and width describe only the peak component — the
        least-squares decomposition IS the background subtraction.

        Peak area in counts is exactly pi * amp * width: summing the
        pixel-integrated Lorentzian amp*w*[arctan((x+.5-c)/w) - arctan((x-.5-c)/w)]
        over all pixels telescopes to amp*w*[arctan(inf) - arctan(-inf)] = amp*w*pi.
        Exact for any width — and exact for 'lorentzian_x_psf' too, because
        the PSF kernel is normalised to unit area (psf.py), and convolution
        with a unit-area kernel conserves the integral. A window sum would be
        WORSE: a +-beta*width window holds only (2/pi)*arctan(beta) of a
        Lorentzian's area (79.5% at beta=3), plus neighbour tails and
        background residue. No frame input on purpose.
        """
        if not fs.is_success:
            return cls(None, None, None, None, None, None)

        left_counts = float(np.pi * fs.left_peak_amplitude * fs.left_peak_width_px)
        right_counts = float(np.pi * fs.right_peak_amplitude * fs.right_peak_width_px)
        e_per_count = electrons_per_count(
            preamp_gain=preamp_gain, emccd_gain=emccd_gain,
            sensitivity_e_per_count=sensitivity_e_per_count)

        return cls(
            left_peak_counts=left_counts,
            right_peak_counts=right_counts,
            left_peak_photons=left_counts * e_per_count,
            right_peak_photons=right_counts * e_per_count,
            total_counts=left_counts + right_counts,
            total_photons=(left_counts + right_counts) * e_per_count,
        )
