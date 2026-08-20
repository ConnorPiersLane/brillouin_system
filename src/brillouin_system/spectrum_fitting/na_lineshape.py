"""
NA-integrated Brillouin lineshape (Mattarelli et al., ACS Photonics 2022,
9, 2087; SI eq. 2s/3s) for CLEAR samples — the ballistic term only.

A high-NA objective collects a cone of scattering angles, so the recorded
peak is a q-weighted superposition of sub-peaks: each collected ray at
deviation angle v from exact backscattering resonates at a frequency
f180 * cos(v/2) (< f180). Fitting a symmetric peak to this asymmetric,
down-shifted blob biases the position low; fitting THIS model returns f180 —
the true 180-degree shift — directly.

The angular collection weight (validated on water):
    W(v) = exp(-2 (v/v0)^2) * sin(v),   v in [0, alpha]
with v0 (effective coupling angular width) and alpha (pupil clip) derived
from the objective geometry there.

Parametrisation notes:
- Works in whatever axis x is given (pixels here). `center` and `rayleigh_px`
  must be in that same axis. Because the NA shift is multiplicative on the
  frequency measured FROM the elastic (Rayleigh) line, each sub-peak sits at
      center - (center - rayleigh_px) * (1 - cos(v/2)),
  so the elastic-line position `rayleigh_px` is a required fixed input.
- Only the peak POSITION is of interest, so the sub-peak core is a Lorentzian
  (near-resonance limit of the DHO); the fitted `gamma` is the intrinsic HWHM,
  the NA broadening is supplied by the fixed kernel.
"""
from __future__ import annotations

import numpy as np

def gaussian_angle_width(
    beam_diameter: float,
    focal_length: float,
    n_sample: float = 1.328,
) -> float:
    """v0: 1/e^2 angular half-width of the Gaussian fiber-coupling weight in
    the SAMPLE, from the collection-mode beam diameter at the objective pupil
    and the objective focal length (air angle refracted into the sample)."""
    theta_air = np.arctan((beam_diameter / 2.0) / focal_length)
    return np.arcsin(np.sin(theta_air) / n_sample)


def pupil_angle_limit(
    pupil_diameter: float,
    focal_length: float,
    n_sample: float = 1.328,
) -> float:
    """alpha: hard aperture clip of the collection cone in the SAMPLE, from
    the physical pupil diameter and the objective focal length."""
    theta_air = np.arctan((pupil_diameter / 2.0) / focal_length)
    return np.arcsin(np.sin(theta_air) / n_sample)


def na_angular_grid(alpha: float, n_quad: int = 41, v0: float | None = None):
    """Return (v, weight, frac_downshift) for the NA collection integral.

    Default (v0=None) is the paper's model (Mattarelli SI): UNIFORM pupil
    transmission, so the only angular weight is the solid angle, sin(v), up to
    a hard aperture cutoff at `alpha` (the effective collection half-angle).
    This has a single geometric input (alpha), no soft coupling parameter.

    Optionally pass v0 to add the Gaussian fiber-coupling apodization
    exp(-2 (v/v0)^2) (validated on water) — used when the
    config sets na_weighting = "uniform_gaussian"; v0 is an empirical, session-drifting
    quantity (the effective fiber-mode diameter), so it must be re-calibrated
    on water per session (na_beam_diameter_mm).

    frac_downshift = 1 - cos(v/2): the fractional shortfall of each sub-peak's
    frequency below the exact-backscattering (180-degree) value.
    """
    v = np.linspace(0.0, float(alpha), int(n_quad))
    weight = np.sin(v)
    if v0 is not None:
        weight = weight * np.exp(-2.0 * (v / v0) ** 2)
    frac = 1.0 - np.cos(v / 2.0)
    return v, weight, frac


def na_mean_shift_ratio(config, n_quad: int = 2001) -> float:
    """Post-hoc scalar <cos(v/2)>: the NA-cone mean of f(v)/f180 under the
    configured collection weight (config.na_weighting).

    This is the paper's post-hoc correction route (Figs. 4/5): fit with the
    standard symmetric model (prm0/prm1, unchanged), then DIVIDE the measured
    shift by this ratio to recover the true 180-degree shift. The integral is a
    pure constant per aperture/weighting — it never enters the fit, and leaves
    split, width and precision untouched (tested 2026-08-05).

        "uniform"          (NA 0.14): ratio ~ 1 - alpha^2/16 -> about +3.5 MHz
                           on water at 5.07 GHz, parameter-free.
        "uniform_gaussian" (NA 0.42): needs na_beam_diameter_mm (the
                           per-session D, calibrated on water) and
                           na_focal_length_mm; a uniform pupil overcorrects
                           at this aperture.

    Reads the na_* fields of the sample find-peaks config.

    na_weighting "none" returns exactly 1.0 (no correction) without touching
    the other na_* fields, so pipelines can divide by this unconditionally.
    """
    weighting = str(getattr(config, "na_weighting", "uniform"))
    if weighting == "none":
        return 1.0
    na = float(config.na_collection)
    n_sample = float(config.na_n_sample)
    if not 0.0 < na < n_sample:
        raise ValueError(
            f"The NA correction requires the collection NA (aperture clip): set "
            f"na_collection (0 < NA < n_sample) in the find-peaks sample config "
            f"(got na_collection={na}, na_n_sample={n_sample})."
        )
    alpha = float(np.arcsin(na / n_sample))
    v0 = None
    if weighting == "uniform_gaussian":
        beam_d = float(config.na_beam_diameter_mm)
        focal = float(config.na_focal_length_mm)
        if beam_d <= 0.0 or focal <= 0.0:
            raise ValueError(
                f"na_weighting 'uniform_gaussian' requires the Gaussian coupling "
                f"geometry: set na_beam_diameter_mm (collection-fiber mode at the "
                f"pupil) and na_focal_length_mm (objective) in the find-peaks "
                f"sample config (got beam={beam_d}, focal={focal})."
            )
        v0 = float(gaussian_angle_width(beam_d, focal, n_sample))
    elif weighting != "uniform":
        raise ValueError(
            f"Unknown na_weighting '{weighting}'. "
            f"Choose 'uniform' or 'uniform_gaussian'."
        )
    v, weight, frac = na_angular_grid(alpha, n_quad, v0=v0)
    numerator = float(np.trapezoid(weight * (1.0 - frac), v))
    denominator = float(np.trapezoid(weight, v))
    return numerator / denominator


# The in-fit NA lineshape builders (make_na_lorentzian and friends) were
# removed 2026-08-20 with the na_lorentzian fitting models: the post-hoc
# scalar above is the production route, validated equivalent on water.
