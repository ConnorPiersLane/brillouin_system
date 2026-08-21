"""Thompson-style localization bound for fitted spectral peaks.

The three-term precision formula of Thompson, Larson & Webb ("Precise
nanometer localization analysis for individual fluorescent probes",
Biophys. J. 82, 2002), applied to a Brillouin peak on a line detector:

    var(center) = F s^2 / N   +   a^2 / 12 / N   +   4 sqrt(pi) s^3 b^2 / (a N^2)
                  photons          pixelation        background

with s the peak HWHM, a the pixel size, N the photons in the peak and b the
rms background noise per pixel.

The GENERIC functions (peak_precision, distance_precision) are unit-agnostic:
s and a share one length unit (the result comes back in that unit), and N and
b share one quantum. theoretical_precision below is the PRODUCTION wrapper:
it reads everything from a fit + calibration and reports MHz.

THIS IS A LOWER BOUND, NOT A PREDICTION of a pipeline's per-frame scatter.
It assumes an ideal (maximum-likelihood) estimator and the noise terms
listed above, nothing else. A real least-squares pipeline sits above it:
x1.14 from the exact Cramer-Rao bound of the full multi-parameter model with
the real noise (read noise, stray-light background), and x1.28 because an
unweighted least-squares fit does not reach the bound -- about x1.45
combined, verified 2026-08-12 by Monte Carlo with exactly known noise
(noise_analysis.monte_carlo_noise_simulation is that tool; scripts in
Data/Calibration_paper_data). Measured per-frame scatter matches x1.45 plus
the ~0.8 MHz per-peak pattern-translation drift, closing the budget with
nothing left over. Use the bound as the floor the measurement cannot beat,
and scale by the measured factor when an absolute prediction is needed.
"""
import math
from dataclasses import dataclass

import numpy as np

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.ccd_characteristics import ccd_config
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.analysis.pixel_counts_and_photons import (
    PixelCountsAndPhotons,
    count_to_electrons,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    sline_from_frame_config,
)
from brillouin_system.spectrum_fitting.psf import detected_hwhm_px
from brillouin_system.spectrum_fitting.spectrum_fitter import is_psf_fit

# Thompson's photon term s^2/N is derived for a GAUSSIAN profile. Brillouin
# peaks are Lorentzian and the HWHM is passed in place of s; a Lorentzian's
# tails carry less information about the centre, so its photon term is
# 2 s^2 / N -- a factor 2 in variance (a Gaussian's factor is 1). Verified
# by Monte Carlo (2000 fits at matched width and photons) and against
# measured scatter: without the factor the formula reads 0.55-0.77x of the
# real per-frame scatter across water, glycerol and cornea; with it,
# 0.78-1.08x.
LORENTZIAN_PHOTON_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Generic, unit-agnostic bound
# ---------------------------------------------------------------------------

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
    n_photons     photons (photoelectrons) in the peak, background-free --
                  see PixelCountsAndPhotons.
    bg_rms        rms background noise per pixel, same quantum as n_photons
                  (shot noise of any background light + read noise, in quadrature).
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
    """
    var = left ** 2 + right ** 2 - 2.0 * correlation * left * right
    return math.sqrt(max(var, 0.0))


# ---------------------------------------------------------------------------
# Production wrapper: fit + calibration in, MHz out
# ---------------------------------------------------------------------------

@dataclass
class TheoreticalPeakStdError:
    """All values in MHz."""
    left_peak_photons_mhz: float | None = None
    left_peak_pixelation_mhz: float | None = None
    left_peak_bg_mhz: float | None = None
    left_peak_total_mhz: float | None = None
    right_peak_photons_mhz: float | None = None
    right_peak_pixelation_mhz: float | None = None
    right_peak_bg_mhz: float | None = None
    right_peak_total_mhz: float | None = None
    # Precision of the peak-distance observable (the one normally reported).
    distance_total_mhz: float | None = None
    # Four-peak standard: per-outer-order totals and the bound of the
    # inverse-variance combined estimator (the variance of the ACTUAL
    # weighted average, using the same photon-term weights combined_shift
    # uses). None unless the fit and the calibration are four-peak.
    outer_left_total_mhz: float | None = None
    outer_right_total_mhz: float | None = None
    combined_total_mhz: float | None = None


def _n_summed_rows(fs: FittedSpectrum) -> int:
    """How many camera rows were summed into this fit's sline.

    From the fit itself (FittedSpectrum.sline_rows — the fitter records the
    band, so the result carries its own acquisition geometry); the live
    sline config is only the fallback for legacy fit objects. The count
    sets the sline's per-pixel read noise, rn*sqrt(n), and scales the
    dark level under the fitted background.
    """
    rows = getattr(fs, "sline_rows", None)
    if rows is not None and len(rows) > 0:
        return len(rows)
    cfg = sline_from_frame_config.get()
    if cfg.row_selection == "auto":
        return int(cfg.n_rows)
    return max(len(cfg.selected_rows), 1)


def theoretical_precision(fs: FittedSpectrum,
                          photons: PixelCountsAndPhotons,
                          calibration_calculator: CalibrationCalculator,
                          preamp_gain: int | float,
                          emccd_gain: int | float,
                          corr_left_right: float = 0.0,
                          ) -> TheoreticalPeakStdError:
    """The Thompson bound of a production fit, in MHz — from ONE frame.

    Inputs are minimal on purpose (user rule 2026-08-20): the FIT — which
    carries its own row band in fs.sline_rows — the calibration, and the
    camera gain settings. Every camera number (read noise, dark level)
    comes from ccd_characteristics; there are no dark-frame inputs — dark
    stacks are not part of the workflow, the measured TOML reference IS
    the dark model.

    NOTE the s convention changed 2026-08-20: s is now the DETECTED photon
    distribution width (fitted core convolved with the camera PSF, pixel
    top-hat excluded), a few percent wider than the fitted gamma for PSF
    fits. The documented pipeline multipliers (x1.14 CRLB, x1.28 LSQ,
    ~x1.45 combined) were calibrated against the gamma-based bound and
    shift by the same few percent.

    Everything the bound needs is derived per frame:
      s  detected HWHM, through the calibration at each peak's own pixel;
      a  the local dispersion there;
      N  background-free peak photons (PixelCountsAndPhotons: the fit
         separates peak from background, and the PSF kernel is unit-area, so
         pi*amp*width is exact);
      b  background NOISE per summed sline pixel, in electrons, from two
         parts in quadrature:
           * read noise — ccd read_noise_counts * sqrt(n_rows): the sline
             sums n_rows camera rows, each carrying the per-pixel read
             noise rms.
           * shot noise of the stray-light background — Poisson on the FITTED
             background level under each peak (fs.*_peak_bg_counts), MINUS
             the dark/bias level (ccd dark_median_counts * n_rows):
             production fits RAW frames, so the fitted background always
             contains that electronic offset, which carries no shot noise.
             Skipping the subtraction inflates the bound ABOVE the measured
             scatter (2026-8-13 300ms_i5: AS bg term 2.66 -> 0.38 MHz,
             total 3.58 -> 2.43 vs measured diff-sd 2.49).
    See the module docstring for what this bound is and is not.
    """
    if not fs.is_success:
        return TheoreticalPeakStdError()

    n_rows = _n_summed_rows(fs)
    ccd = ccd_config.get()
    dark_counts = ccd.dark_median_counts * n_rows

    calc = calibration_calculator

    a_l = abs(calc.df_left_peak(px=fs.left_peak_center_px, dpx=1))
    a_r = abs(calc.df_right_peak(px=fs.right_peak_center_px, dpx=1))

    # s is the width of the photon distribution AS DETECTED, not the fitted
    # Lorentzian core: for a PSF-convolved fit the fitted gamma is the core
    # BEFORE the camera PSF, but the photons arrive spread by
    # Lorentzian (x) Gauss(sigma) (x) tail(tau) — a few-to-ten percent wider
    # at production widths. The pixel top-hat stays OUT of s (it is the
    # separate a^2/12 pixelation term). For a plain-Lorentzian fit the
    # fitted width already IS the detected width.
    if is_psf_fit(fs.model):
        k = sline_from_frame_config.get()
        w_l = detected_hwhm_px(fs.left_peak_width_px,
                               k.psf_sigma_px, k.psf_tau_left_px)
        w_r = detected_hwhm_px(fs.right_peak_width_px,
                               k.psf_sigma_px, k.psf_tau_right_px)
    else:
        w_l, w_r = fs.left_peak_width_px, fs.right_peak_width_px
    s_l = a_l * float(w_l)
    s_r = a_r * float(w_r)

    read_per_sline_px = ccd.read_noise_counts * math.sqrt(n_rows)

    def b_electrons(read_counts, fitted_bg_counts):
        read_e = count_to_electrons(read_counts or 0.0,
                                    preamp_gain=preamp_gain,
                                    emccd_gain=emccd_gain)
        # Poisson: the background light's variance in electrons equals its level —
        # counting only the LIGHT part (bias is an offset, no shot noise).
        light_counts = max((fitted_bg_counts or 0.0) - dark_counts, 0.0)
        bg_light_var_e = count_to_electrons(light_counts,
                                            preamp_gain=preamp_gain,
                                            emccd_gain=emccd_gain)
        return math.sqrt(read_e ** 2 + bg_light_var_e)

    b_l = b_electrons(read_per_sline_px, fs.left_peak_bg_counts)
    b_r = b_electrons(read_per_sline_px, fs.right_peak_bg_counts)

    left = peak_precision(width=s_l, n_photons=photons.left_peak_photons,
                          bg_rms=b_l, pixel_size=a_l)
    right = peak_precision(width=s_r, n_photons=photons.right_peak_photons,
                           bg_rms=b_r, pixel_size=a_r)

    # Distance observable: per-peak frequency errors back to pixels through
    # each order's own slope, combined, then out through the distance track.
    a_d = abs(calc.df_peak_distance(px=fs.inter_peak_distance, dpx=1))
    dx_d_total = a_d * distance_precision(
        left.total / a_l, right.total / a_r, correlation=corr_left_right)

    # Four-peak standard: bounds for the outer orders through their own
    # tracks, and the variance of the ACTUAL combined estimator — the
    # weighted average with combined_shift's photon-term weights:
    # var = sum(w_i^2 sigma_i^2) with sum(w_i) = 1.
    outer_left = outer_right = None
    combined_total = None
    four_peak = calc.combined_shift(fs)
    if four_peak is not None and photons.outer_left_peak_photons:
        a_ol = abs(calc.df_outer_left_peak(px=fs.outer_left_peak_center_px, dpx=1))
        a_or = abs(calc.df_outer_right_peak(px=fs.outer_right_peak_center_px, dpx=1))
        if is_psf_fit(fs.model):
            k = sline_from_frame_config.get()
            w_ol = detected_hwhm_px(fs.outer_left_peak_width_px,
                                    k.psf_sigma_px, k.psf_tau_outer_left_px)
            w_or = detected_hwhm_px(fs.outer_right_peak_width_px,
                                    k.psf_sigma_px, k.psf_tau_outer_right_px)
        else:
            w_ol = fs.outer_left_peak_width_px
            w_or = fs.outer_right_peak_width_px
        outer_left = peak_precision(
            width=a_ol * float(w_ol), n_photons=photons.outer_left_peak_photons,
            bg_rms=b_electrons(read_per_sline_px, fs.outer_left_peak_bg_counts),
            pixel_size=a_ol)
        outer_right = peak_precision(
            width=a_or * float(w_or), n_photons=photons.outer_right_peak_photons,
            bg_rms=b_electrons(read_per_sline_px, fs.outer_right_peak_bg_counts),
            pixel_size=a_or)
        sigmas = (outer_left.total, left.total, right.total, outer_right.total)
        combined_total = math.sqrt(sum(
            w * w * s * s for w, s in zip(four_peak.weights, sigmas)))

    return TheoreticalPeakStdError(
        left_peak_photons_mhz=left.photons * 1000,
        left_peak_pixelation_mhz=left.pixelation * 1000,
        left_peak_bg_mhz=left.background * 1000,
        left_peak_total_mhz=left.total * 1000,
        right_peak_photons_mhz=right.photons * 1000,
        right_peak_pixelation_mhz=right.pixelation * 1000,
        right_peak_bg_mhz=right.background * 1000,
        right_peak_total_mhz=right.total * 1000,
        distance_total_mhz=dx_d_total * 1000,
        outer_left_total_mhz=(outer_left.total * 1000
                              if outer_left is not None else None),
        outer_right_total_mhz=(outer_right.total * 1000
                               if outer_right is not None else None),
        combined_total_mhz=(combined_total * 1000
                            if combined_total is not None else None),
    )
