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
the real noise (read noise, stray-light pedestal), and x1.28 because an
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
from brillouin_system.spectrum_fitting.noise_analysis.pixel_counts_and_photons import (
    PixelCountsAndPhotons,
    count_to_electrons,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    psf_config,
    sline_from_frame_config,
)
from brillouin_system.spectrum_fitting.psf import detected_hwhm_px
from brillouin_system.spectrum_fitting.spectrum_fitter import is_psf_fit

# Thompson's photon term s^2/N is derived for a GAUSSIAN profile. Brillouin
# peaks are Lorentzian and the HWHM is passed in place of s; a Lorentzian's
# tails carry less information about the centre, so its photon term is
# 2 s^2 / N -- a factor 2 in variance. Verified by Monte Carlo (2000 fits at
# matched width and photons) and against measured scatter: without the
# factor the formula reads 0.55-0.77x of the real per-frame scatter across
# water, glycerol and cornea; with it, 0.78-1.08x.
LORENTZIAN_PHOTON_FACTOR = 2.0
GAUSSIAN_PHOTON_FACTOR = 1.0

# Per-pixel read noise rms [counts] (4.3 e-) — measured value and
# provenance live in ccd_characteristics.toml (dark exposure ladder
# 2026-08-19; a closed-shutter frame's per-pixel std IS the read noise, no
# measurable dark current). A per-scan dark stack taken at the live
# settings is preferred over the constant (readout-mode dependence).
# Import-time alias for backwards compatibility; theoretical_precision
# reads the live config.
READ_NOISE_COUNTS = ccd_config.get().read_noise_counts


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
    """ All Values in MHz"""
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


def get_b_values(std_img, fit, k: float = 2.0,
                 rows: list[int] | None = None,
                 ) -> tuple[float | None, float | None] | None:
    """Per-sline-pixel noise std near the left/right peaks, from a std frame.

    std_img is a per-pixel std image (production: the scan's closed-shutter
    dark stack, whose std is the read noise); it is combined across the
    summed rows in quadrature, median inside a window of +-k*width around
    each fitted peak.

    rows: the rows actually summed into the fitted sline. Pass them when
    known (a scan's stored band can differ from the live config) — the
    global sline config is only the fallback.
    """
    if std_img is None:
        return None
    if not fit.is_success:
        return None, None

    H, W = std_img.shape

    # Select rows (same as in get_px_sline_from_image)
    if rows is None:
        rows = sline_from_frame_config.get().selected_rows
    if not rows or not all(0 <= r < H for r in rows):
        print("[get_b_values] Warning: Invalid or empty row list — using full image height.")
        rows = list(range(H))

    # The signal sums rows, so the noise combines in quadrature.
    binned_std_full = np.sqrt(np.sum(std_img[rows, :] ** 2, axis=0))
    px_full = np.arange(W)

    def side_median_b(center: float, width: float) -> float | None:
        if center is None or width is None:
            return None
        center = int(round(center))
        halfwin = int(math.ceil(k * float(width)))
        lo_idx = max(0, center - halfwin)
        hi_idx = min(len(px_full), center + halfwin)
        mask = np.zeros_like(binned_std_full, dtype=bool)
        mask[lo_idx:hi_idx] = True
        if not np.any(mask):
            return None
        return float(np.median(binned_std_full[mask]))

    left_b = side_median_b(fit.left_peak_center_px, fit.left_peak_width_px)
    right_b = side_median_b(fit.right_peak_center_px, fit.right_peak_width_px)
    return left_b, right_b


def _n_summed_rows(rows: list[int] | None = None) -> int:
    """Rows summed into the sline, for the read-noise fallback."""
    if rows is not None:
        return max(len(rows), 1)
    cfg = sline_from_frame_config.get()
    if cfg.row_selection == "auto":
        return int(cfg.n_rows)
    return max(len(cfg.selected_rows), 1)


def theoretical_precision(fs: FittedSpectrum,
                          photons: PixelCountsAndPhotons,
                          calibration_calculator: CalibrationCalculator,
                          dark_frame_std: np.ndarray | None,
                          preamp_gain: int | float,
                          emccd_gain: int | float,
                          corr_left_right: float = 0.0,
                          pedestal_bias_counts: float = 0.0,
                          sline_rows: list[int] | None = None,
                          ) -> TheoreticalPeakStdError:
    """The Thompson bound of a production fit, in MHz — from ONE frame.

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
         separates peak from pedestal, and the PSF kernel is unit-area, so
         pi*amp*width is exact);
      b  background NOISE per summed sline pixel, in electrons, from two
         parts in quadrature:
           * read noise — the scan's closed-shutter dark stack: its
             per-pixel std IS the read noise (no dark current), summed over
             the rows in quadrature by get_b_values. Falls back to the
             measured READ_NOISE_COUNTS * sqrt(n_rows) when no dark stack
             travels with the data (dark_frame_std=None).
           * shot noise of the stray-light pedestal — Poisson on the FITTED
             background level under each peak (fs.*_peak_bg_counts).

    pedestal_bias_counts: the camera dark/bias level per summed sline pixel.
    Production fits RAW frames (nothing subtracted from the data, user rule
    2026-08-20), so the fitted background always contains this level
    (~200 counts/px x n rows on this camera). It is an electronic offset,
    not light — it carries no shot noise — so it is removed from the
    pedestal HERE, analytically, before the Poisson term. Callers pass the
    scan's own dark-stack median x summed rows (or a frame-median estimate
    when no darks were taken). Without it the bound is inflated ABOVE the
    measured scatter (verified on 2026-8-13 300ms_i5: AS bg term
    2.66 -> 0.38 MHz, total 3.58 -> 2.43 vs measured diff-sd 2.49).
    See the module docstring for what this bound is and is not.
    """
    if not fs.is_success:
        return TheoreticalPeakStdError()

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
        k = psf_config.get()
        w_l = detected_hwhm_px(fs.left_peak_width_px,
                               k.psf_sigma_px, k.psf_tau_left_px)
        w_r = detected_hwhm_px(fs.right_peak_width_px,
                               k.psf_sigma_px, k.psf_tau_right_px)
    else:
        w_l, w_r = fs.left_peak_width_px, fs.right_peak_width_px
    s_l = a_l * float(w_l)
    s_r = a_r * float(w_r)

    if dark_frame_std is None:
        fallback = (ccd_config.get().read_noise_counts
                    * math.sqrt(_n_summed_rows(sline_rows)))
        read_counts_l, read_counts_r = fallback, fallback
    else:
        read_counts_l, read_counts_r = get_b_values(std_img=dark_frame_std,
                                                    fit=fs, rows=sline_rows)

    def b_electrons(read_counts, pedestal_counts):
        read_e = count_to_electrons(read_counts or 0.0,
                                    preamp_gain=preamp_gain,
                                    emccd_gain=emccd_gain)
        # Poisson: the pedestal's variance in electrons equals its level —
        # counting only the LIGHT part (bias is an offset, no shot noise).
        light_counts = max((pedestal_counts or 0.0) - pedestal_bias_counts, 0.0)
        pedestal_var_e = count_to_electrons(light_counts,
                                            preamp_gain=preamp_gain,
                                            emccd_gain=emccd_gain)
        return math.sqrt(read_e ** 2 + pedestal_var_e)

    b_l = b_electrons(read_counts_l, fs.left_peak_bg_counts)
    b_r = b_electrons(read_counts_r, fs.right_peak_bg_counts)

    left = peak_precision(width=s_l, n_photons=photons.left_peak_photons,
                          bg_rms=b_l, pixel_size=a_l)
    right = peak_precision(width=s_r, n_photons=photons.right_peak_photons,
                           bg_rms=b_r, pixel_size=a_r)

    # Distance observable: per-peak frequency errors back to pixels through
    # each order's own slope, combined, then out through the distance track.
    a_d = abs(calc.df_peak_distance(px=fs.inter_peak_distance, dpx=1))
    dx_d_total = a_d * distance_precision(
        left.total / a_l, right.total / a_r, correlation=corr_left_right)

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
    )
