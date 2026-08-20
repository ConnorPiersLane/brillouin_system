"""Thompson-style localization bound for fitted spectral peaks.

The three-term precision formula of Thompson, Larson & Webb ("Precise
nanometer localization analysis for individual fluorescent probes",
Biophys. J. 82, 2002), applied to a Brillouin peak on a line detector:

    var(center) = F s^2 / N   +   a^2 / 12 / N   +   4 sqrt(pi) s^3 b^2 / (a N^2)
                  photons          pixelation        background

with s the peak HWHM, a the pixel size, N the photons in the peak and b the
rms background noise per pixel.

Units are the caller's, with one rule: s and a share one length unit (the
result comes back in that unit), and N and b share one quantum (photons or
electrons -- consistently). The two standard usages:

* pixel domain: s in px, a = 1. Multiply the result by the local dispersion
  [GHz/px] of the observable's own calibration track to quote GHz.
* frequency domain: s in GHz, a = local dispersion [GHz/px]; result in GHz.

THIS IS A LOWER BOUND, NOT A PREDICTION of a pipeline's per-frame scatter.
It assumes an ideal (maximum-likelihood) estimator and the noise terms
listed above, nothing else. A real least-squares pipeline sits above it:
for the instrument in the paper, x1.14 from the exact Cramer-Rao bound of
the full multi-parameter model with the real noise (read noise, stray-light
pedestal), and x1.28 because an unweighted least-squares fit does not reach
the bound -- about x1.45 combined, verified by Monte Carlo with exactly
known noise (noise_analysis.monte_carlo_frames is that tool). Use the bound
as the floor a measurement cannot beat, and scale by a measured factor when
an absolute prediction is needed.
"""
import math
from dataclasses import dataclass
from typing import Sequence

# Thompson's photon term s^2/N is derived for a GAUSSIAN profile. Brillouin
# peaks are Lorentzian and the HWHM is passed in place of s; a Lorentzian's
# tails carry less information about the centre, so its photon term is
# 2 s^2 / N -- a factor 2 in variance. Verified by Monte Carlo (2000 fits at
# matched width and photons) and against measured scatter: without the
# factor the formula reads 0.55-0.77x of the real per-frame scatter across
# water, glycerol and cornea; with it, 0.78-1.08x.
LORENTZIAN_PHOTON_FACTOR = 2.0
GAUSSIAN_PHOTON_FACTOR = 1.0


def peak_photons(amplitude: float, width: float,
                 gain_e_per_count: float) -> float:
    """Actual photons (photoelectrons) N in a fitted Lorentzian peak.

    The area of the pixel-integrated Lorentzian is exactly pi*amp*width --
    summing amp*w*[arctan((x+.5-c)/w) - arctan((x-.5-c)/w)] over all pixels
    telescopes to amp*w*pi. Exact for any width, so narrow peaks need no
    correction. Hence

        N = gain * pi * amplitude[counts] * width[px].

    amplitude and width come straight from the fit (counts, px HWHM);
    gain_e_per_count is the digitised sensitivity with any preamp multiplier
    already folded in. In EM mode divide N by the excess-noise factor (~2)
    before feeding a bound -- the EM register multiplies stochastically.
    """
    if amplitude < 0.0 or width <= 0.0 or gain_e_per_count <= 0.0:
        raise ValueError("amplitude must be >= 0, width and gain > 0.")
    return gain_e_per_count * math.pi * amplitude * width


@dataclass
class PeakPrecision:
    """Std of one peak-centre estimate, per noise source and total.

    Same length unit as the s / a inputs that produced it.
    """
    photons: float
    pixelation: float
    background: float
    total: float


def peak_precision(
    width: float,
    n_photons: float,
    bg_rms: float = 0.0,
    pixel_size: float = 1.0,
    photon_factor: float = LORENTZIAN_PHOTON_FACTOR,
) -> PeakPrecision:
    """Thompson bound on one peak centre.

    width         peak HWHM (s), in the caller's length unit. The RAW fitted
                  width as the peak lands on the detector -- still instrument
                  broadened. The bound is set by what is measured, not by the
                  sample's own linewidth.
    n_photons     actual photons (photoelectrons) in the peak, background-free
                  -- see peak_photons().
    bg_rms        rms background noise per pixel, same quantum as n_photons
                  (shot noise of any pedestal + read noise, in quadrature).
                  0 drops the background term.
    pixel_size    a, in the same length unit as width (1.0 in pixel domain).
    photon_factor 2.0 for Lorentzian peaks (default), 1.0 for Gaussian.
    """
    if width <= 0.0 or n_photons <= 0.0 or pixel_size <= 0.0:
        raise ValueError("width, n_photons and pixel_size must be > 0.")
    s, a, n, b = float(width), float(pixel_size), float(n_photons), float(bg_rms)

    var_photons = photon_factor * s ** 2 / n
    var_pixelation = a ** 2 / 12.0 / n
    var_background = 4.0 * math.sqrt(math.pi) * s ** 3 * b ** 2 / (a * n ** 2)

    return PeakPrecision(
        photons=math.sqrt(var_photons),
        pixelation=math.sqrt(var_pixelation),
        background=math.sqrt(var_background),
        total=math.sqrt(var_photons + var_pixelation + var_background),
    )


def peaks_precision(
    widths: Sequence[float],
    n_photons: Sequence[float],
    bg_rms: float | Sequence[float] = 0.0,
    pixel_size: float | Sequence[float] = 1.0,
    photon_factor: float = LORENTZIAN_PHOTON_FACTOR,
) -> list[PeakPrecision]:
    """Per-peak bounds for a multi-peak spectrum, ordered like the inputs.

    Works for the main pair (2 peaks) and for the 4-peak fit with the outer
    orders alike -- pass one width and one photon number per peak, left to
    right. bg_rms and pixel_size may be scalars (shared) or per-peak
    sequences; a per-peak pixel_size is how different local dispersions of
    the four calibration tracks enter in the frequency domain.
    """
    n = len(widths)
    if len(n_photons) != n:
        raise ValueError("widths and n_photons must have one entry per peak.")

    def per_peak(value, name):
        if isinstance(value, (int, float)):
            return [float(value)] * n
        if len(value) != n:
            raise ValueError(f"{name} must be a scalar or one entry per peak.")
        return [float(v) for v in value]

    bgs = per_peak(bg_rms, "bg_rms")
    pixels = per_peak(pixel_size, "pixel_size")
    return [
        peak_precision(width=w, n_photons=p, bg_rms=b, pixel_size=a,
                       photon_factor=photon_factor)
        for w, p, b, a in zip(widths, n_photons, bgs, pixels)
    ]


def distance_precision(
    left: float,
    right: float,
    correlation: float = 0.0,
) -> float:
    """Bound on the inter-peak distance from the two per-peak bounds.

    d = c_right - c_left, so var(d) = var(l) + var(r) - 2 rho sd(l) sd(r).

    correlation is between the two CENTRE ERRORS, and 0 is the right choice
    for a shot-noise bound: the two peaks are made of different photons on
    different pixels. (A small correlation seen in repeated measurements is
    common-mode drift, not photon noise -- do not fold it into the bound.)

    Give both inputs in one unit; the result is in that unit. When the two
    peaks sit on calibration tracks with different local dispersion, convert
    each per-peak bound to pixels through its own track first, combine here,
    then apply the distance track's dispersion.
    """
    var = left ** 2 + right ** 2 - 2.0 * correlation * left * right
    return math.sqrt(max(var, 0.0))
