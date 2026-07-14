"""
NA-integrated Brillouin lineshape (Mattarelli et al., ACS Photonics 2022,
9, 2087; SI eq. 2s/3s) for CLEAR samples — the ballistic term only.

A high-NA objective collects a cone of scattering angles, so the recorded
peak is a q-weighted superposition of sub-peaks: each collected ray at
deviation angle v from exact backscattering resonates at a frequency
f180 * cos(v/2) (< f180). Fitting a symmetric peak to this asymmetric,
down-shifted blob biases the position low; fitting THIS model returns f180 —
the true 180-degree shift — directly.

The angular collection weight reuses na_correction5 (validated on water):
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

from brillouin_system.spectrum_fitting.na_correction5 import (
    gaussian_angle_width,
    pupil_angle_limit,
)


def na_angular_grid(alpha: float, n_quad: int = 41, v0: float | None = None):
    """Return (v, weight, frac_downshift) for the NA collection integral.

    Default (v0=None) is the paper's model (Mattarelli SI): UNIFORM pupil
    transmission, so the only angular weight is the solid angle, sin(v), up to
    a hard aperture cutoff at `alpha` (the effective collection half-angle).
    This has a single geometric input (alpha), no soft coupling parameter.

    Optionally pass v0 to add the Gaussian fiber-coupling apodization
    exp(-2 (v/v0)^2) (na_correction5 form) — NOT recommended for the fit model
    because v0 is an empirical, session-drifting quantity.

    frac_downshift = 1 - cos(v/2): the fractional shortfall of each sub-peak's
    frequency below the exact-backscattering (180-degree) value.
    """
    v = np.linspace(0.0, float(alpha), int(n_quad))
    weight = np.sin(v)
    if v0 is not None:
        weight = weight * np.exp(-2.0 * (v / v0) ** 2)
    frac = 1.0 - np.cos(v / 2.0)
    return v, weight, frac


def make_na_lorentzian(rayleigh_px, alpha, n_quad: int = 41, v0: float | None = None):
    """
    Build an NA-integrated Lorentzian model anchored at the elastic line.

    Fixed inputs: rayleigh_px (elastic-line pixel) and alpha (effective
    collection half-angle in the sample; e.g. na_correction5.pupil_angle_limit,
    or an effective value calibrated on a reference liquid). By default the
    pupil is treated as UNIFORM (paper model) — no Gaussian coupling.

    Free parameters:  [amp, center, gamma, offset]
        center = the 180-degree peak position (px) -> the true shift downstream
        gamma  = intrinsic HWHM (px); NA broadening comes from the fixed kernel
    """
    R = float(rayleigh_px)
    v, weight, frac = na_angular_grid(alpha, n_quad, v0=v0)
    wsum = float(np.trapezoid(weight, v))
    if wsum <= 0:
        raise ValueError("NA collection weight integrates to <= 0.")

    def _na_lorentzian(x, amp, center, gamma, offset):
        x = np.asarray(x, dtype=float)
        gamma = max(float(gamma), 1e-9)
        # Sub-peak centers: down-shifted from `center` by the NA fraction of the
        # elastic-line distance (center - R).
        sub_centers = center - (center - R) * frac          # (n_quad,)
        dx = x[:, None] - sub_centers[None, :]              # (nx, n_quad)
        lor = gamma**2 / (dx**2 + gamma**2)                 # (nx, n_quad)
        integ = np.trapezoid(weight[None, :] * lor, v, axis=1) / wsum
        return amp * integ + offset

    return _na_lorentzian


def make_na_lorentzian_from_geometry(
    rayleigh_px,
    pupil_diameter,
    focal_length,
    n_sample: float = 1.328,
    n_quad: int = 41,
    beam_diameter: float | None = None,
):
    """
    Convenience: derive the aperture `alpha` from the objective pupil geometry
    (na_correction5.pupil_angle_limit). Uniform pupil by default; pass
    beam_diameter to additionally apply the Gaussian coupling apodization.
    """
    alpha = pupil_angle_limit(pupil_diameter, focal_length, n_sample)
    v0 = None
    if beam_diameter is not None:
        v0 = gaussian_angle_width(beam_diameter, focal_length, n_sample)
    return make_na_lorentzian(rayleigh_px, alpha, n_quad, v0=v0)
