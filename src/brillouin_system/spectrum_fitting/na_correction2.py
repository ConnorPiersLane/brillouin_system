"""
Brillouin peak shift from untruncated Gaussian illumination and finite collection NA
------------------------------------------------------------------------------------

Model assumptions
-----------------
1) Backscattering geometry.
2) Incoming beam is an untruncated Gaussian in angle.
3) Incoming Gaussian is specified by:
       - beam_diameter_in : 1/e^2 beam diameter at the pupil/lens plane
       - focal_length_in  : focal length of the input focusing optics
4) Collection is uniform over the accepted output cone.
5) No multiple scattering.
6) Optional full Lorentzian-broadened spectrum.

Key formulas
------------
Incoming Gaussian angular width:
    u0 = arctan((beam_diameter_in / 2) / focal_length_in)

Collection half-angle:
    alpha_out = arcsin(na_out / n_ang)

Ray-pair Brillouin shift:
    f_B(u, v, dphi) / f180
      = sqrt((1 + cos(u)cos(v) + sin(u)sin(v)cos(dphi)) / 2)

Incoming weight:
    W_i(u) = exp[-2 * (u/u0)^2] * sin(u)

Collection weight:
    W_c(v) = sin(v)

Exact mean shift ratio:
    <f_B>/f180 = ∫∫∫ W_i W_c (f_B/f180) du dv dphi / ∫∫∫ W_i W_c du dv dphi

Notes
-----
- n_ang is used only to convert output NA to an internal angular limit.
- There is no input NA in this untruncated Gaussian model.
- If you want an effective model that matches data better, you may choose n_ang = 1.
- Frequencies may be in any unit as long as f180, gamma, and f_axis use the same unit.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# 1. Basic helpers
# ---------------------------------------------------------------------

def internal_half_angle(na: float, n_ang: float) -> float:
    """
    Convert an external NA into an internal half-angle.

    Parameters
    ----------
    na : float
        External numerical aperture.
    n_ang : float
        Refractive-index-like angular scaling parameter.
        For a physical internal-angle model, use n_ang = n_sample.
        For an effective external-angle model, you may set n_ang = 1.

    Returns
    -------
    float
        Internal half-angle in radians:
            alpha = arcsin(na / n_ang)
    """
    if na < 0:
        raise ValueError("na must be non-negative.")
    if n_ang <= 0:
        raise ValueError("n_ang must be positive.")

    ratio = na / n_ang
    if ratio > 1:
        raise ValueError(
            f"na / n_ang = {ratio:.6f} > 1, which is not physical."
        )

    return np.arcsin(ratio)


def gaussian_input_angular_width(
    beam_diameter_in: float,
    focal_length_in: float,
) -> float:
    """
    Exact angular width parameter of the incoming Gaussian beam.

    Parameters
    ----------
    beam_diameter_in : float
        1/e^2 beam diameter at the pupil/lens plane.
    focal_length_in : float
        Focal length of the input focusing optics.

    Returns
    -------
    float
        Angular Gaussian width u0 in radians:
            u0 = arctan((beam_diameter_in / 2) / focal_length_in)
    """
    if beam_diameter_in <= 0:
        raise ValueError("beam_diameter_in must be > 0.")
    if focal_length_in <= 0:
        raise ValueError("focal_length_in must be > 0.")

    return np.arctan((beam_diameter_in / 2.0) / focal_length_in)


def gaussian_illumination_weight(u: np.ndarray, u0: float) -> np.ndarray:
    """
    Angular weighting of the untruncated incoming Gaussian beam.

    W_i(u) = exp[-2 * (u/u0)^2] * sin(u)
    """
    if u0 <= 0:
        raise ValueError("u0 must be > 0.")
    return np.exp(-2.0 * (u / u0) ** 2) * np.sin(u)


def uniform_collection_weight(v: np.ndarray) -> np.ndarray:
    """
    Angular weighting of the collected light.

    W_c(v) = sin(v)
    """
    return np.sin(v)


def brillouin_shift_ratio_exact(
    u: np.ndarray,
    v: np.ndarray,
    dphi: np.ndarray,
) -> np.ndarray:
    """
    Compute exact f_B / f180 for each ray-pair geometry.

    Geometry
    --------
    u    : incoming polar angle from +z
    v    : outgoing polar angle from -z
    dphi : relative azimuth

    The scattering angle theta satisfies:
        cos(theta) = -cos(u)cos(v) - sin(u)sin(v)cos(dphi)

    Therefore:
        f_B / f180 = sin(theta/2)
                   = sqrt((1 - cos(theta))/2)
                   = sqrt((1 + cos(u)cos(v) + sin(u)sin(v)cos(dphi))/2)
    """
    arg = 0.5 * (
        1.0
        + np.cos(u) * np.cos(v)
        + np.sin(u) * np.sin(v) * np.cos(dphi)
    )

    # Numerical guard
    arg = np.clip(arg, 0.0, None)
    return np.sqrt(arg)


# ---------------------------------------------------------------------
# 2. Exact mean shift ratio
# ---------------------------------------------------------------------

def mean_shift_ratio_exact_untruncated_gaussian_input(
    beam_diameter_in: float,
    focal_length_in: float,
    na_out: float,
    n_ang: float,
    n_u: int = 240,
    n_v: int = 180,
    n_phi: int = 180,
    gaussian_cutoff_sigma: float = 1.1,
) -> float:
    """
    Exact weighted mean Brillouin shift ratio <f_B>/f180 with:

    - untruncated Gaussian illumination
    - uniform collection
    - no small-angle approximation

    Parameters
    ----------
    beam_diameter_in : float
        1/e^2 diameter of the input Gaussian beam at the pupil/lens plane.
    focal_length_in : float
        Focal length of the input focusing optics.
    na_out : float
        External collection NA.
    n_ang : float
        Refractive-index-like parameter used to convert output NA to internal angle.
    n_u, n_v, n_phi : int
        Integration grid sizes.
    gaussian_cutoff_sigma : float, default=5.0
        Practical upper integration cutoff for the untruncated Gaussian.
        The illumination integral is carried from 0 to u_max = gaussian_cutoff_sigma * u0.

    Returns
    -------
    float
        Mean shift ratio <f_B>/f180.
    """
    u0 = gaussian_input_angular_width(beam_diameter_in, focal_length_in)
    u_max = gaussian_cutoff_sigma * u0

    alpha_out = internal_half_angle(na_out, n_ang)

    # Integration grids
    u = np.linspace(0.0, u_max, n_u)
    v = np.linspace(0.0, alpha_out, n_v)
    dphi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    # Broadcasted arrays
    U = u[:, None, None]
    V = v[None, :, None]
    P = dphi[None, None, :]

    # Angular weights
    Wi = gaussian_illumination_weight(U, u0)
    Wc = uniform_collection_weight(V)
    W = Wi * Wc

    # Exact shift ratio
    ratio = brillouin_shift_ratio_exact(U, V, P)

    numerator = np.trapezoid(
        np.trapezoid(
            np.trapezoid(W * ratio, dphi, axis=2),
            v,
            axis=1,
        ),
        u,
        axis=0,
    )

    denominator = (2.0 * np.pi) * np.trapezoid(
        np.trapezoid(W[..., 0], v, axis=1),
        u,
        axis=0,
    )

    return float(numerator / denominator)


# ---------------------------------------------------------------------
# 3. Absolute shift
# ---------------------------------------------------------------------

def na_shift_absolute_untruncated_gaussian_input(
    f180: float,
    beam_diameter_in: float,
    focal_length_in: float,
    na_out: float,
    n_ang: float,
    n_u: int = 240,
    n_v: int = 180,
    n_phi: int = 180,
    gaussian_cutoff_sigma: float = 5.0,
) -> dict:
    """
    Compute the absolute NA-induced peak shift.

    Parameters
    ----------
    f180 : float
        Nominal 180-degree Brillouin shift (same frequency unit as desired output).
    beam_diameter_in : float
        1/e^2 diameter of the input Gaussian beam at the pupil/lens plane.
    focal_length_in : float
        Focal length of the input focusing optics.
    na_out : float
        External collection NA.
    n_ang : float
        Refractive-index-like parameter used for output angle conversion.
    n_u, n_v, n_phi : int
        Integration grid sizes.
    gaussian_cutoff_sigma : float, default=5.0
        Practical Gaussian integration cutoff in units of u0.

    Returns
    -------
    dict
        Dictionary containing:
        - 'f180'
        - 'mean_ratio'
        - 'mean_shift'
        - 'delta_f_na'
        - 'u0_rad'
        - 'u0_deg'
    """
    u0 = gaussian_input_angular_width(beam_diameter_in, focal_length_in)

    ratio = mean_shift_ratio_exact_untruncated_gaussian_input(
        beam_diameter_in=beam_diameter_in,
        focal_length_in=focal_length_in,
        na_out=na_out,
        n_ang=n_ang,
        n_u=n_u,
        n_v=n_v,
        n_phi=n_phi,
        gaussian_cutoff_sigma=gaussian_cutoff_sigma,
    )

    mean_shift = ratio * f180
    delta_f_na = mean_shift - f180

    return {
        "f180": float(f180),
        "mean_ratio": float(ratio),
        "mean_shift": float(mean_shift),
        "delta_f_na": float(delta_f_na),
        "u0_rad": float(u0),
        "u0_deg": float(np.degrees(u0)),
    }


# ---------------------------------------------------------------------
# 4. Full Lorentzian spectrum
# ---------------------------------------------------------------------

def brillouin_spectrum_lorentzian_untruncated_gaussian_input(
    f_axis: np.ndarray,
    f180: float,
    gamma: float,
    beam_diameter_in: float,
    focal_length_in: float,
    na_out: float,
    n_ang: float,
    n_u: int = 180,
    n_v: int = 140,
    n_phi: int = 140,
    gaussian_cutoff_sigma: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the full Lorentzian-broadened Brillouin spectrum for:

    - untruncated Gaussian illumination
    - uniform collection
    - exact ray-pair geometry

    Spectrum:
        S(f) = ∫∫∫ W_i(u) W_c(v)
               * gamma / [(f - f_B(u,v,dphi))^2 + gamma^2]
               du dv dphi

    Parameters
    ----------
    f_axis : np.ndarray
        Frequency axis where the spectrum is evaluated.
    f180 : float
        Nominal 180-degree Brillouin shift.
    gamma : float
        Lorentzian half-width.
    beam_diameter_in : float
        1/e^2 diameter of the input Gaussian beam at the pupil/lens plane.
    focal_length_in : float
        Focal length of the input focusing optics.
    na_out : float
        External collection NA.
    n_ang : float
        Refractive-index-like parameter used for output angle conversion.
    n_u, n_v, n_phi : int
        Integration grid sizes.
    gaussian_cutoff_sigma : float, default=5.0
        Practical Gaussian cutoff in units of u0.
    normalize : bool, default=True
        If True, normalize the spectrum maximum to 1.

    Returns
    -------
    np.ndarray
        Spectrum evaluated on f_axis.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    u0 = gaussian_input_angular_width(beam_diameter_in, focal_length_in)
    u_max = gaussian_cutoff_sigma * u0
    alpha_out = internal_half_angle(na_out, n_ang)

    # Integration grids
    u = np.linspace(0.0, u_max, n_u)
    v = np.linspace(0.0, alpha_out, n_v)
    dphi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    U = u[:, None, None]
    V = v[None, :, None]
    P = dphi[None, None, :]

    Wi = gaussian_illumination_weight(U, u0)
    Wc = uniform_collection_weight(V)
    W = Wi * Wc

    ratio = brillouin_shift_ratio_exact(U, V, P)
    f_centers = f180 * ratio

    spectrum = np.zeros_like(f_axis, dtype=float)

    for i, f in enumerate(f_axis):
        lorentz = gamma / ((f - f_centers) ** 2 + gamma**2)
        spectrum[i] = np.trapezoid(
            np.trapezoid(
                np.trapezoid(W * lorentz, dphi, axis=2),
                v,
                axis=1,
            ),
            u,
            axis=0,
        )

    if normalize and spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()

    return spectrum


# ---------------------------------------------------------------------
# 5. Optional helper: effective Gaussian input NA
# ---------------------------------------------------------------------

def gaussian_input_effective_na(
    beam_diameter_in: float,
    focal_length_in: float,
    n_ang: float = 1.0,
) -> float:
    """
    Return a characteristic effective NA for the untruncated Gaussian input beam.

    This is not a hard cutoff NA. It is based on the Gaussian angular width:
        u0 = arctan((D/2)/f)
        NA_eff = n_ang * sin(u0)

    Parameters
    ----------
    beam_diameter_in : float
        1/e^2 beam diameter at the pupil/lens plane.
    focal_length_in : float
        Focal length of the input focusing optics.
    n_ang : float, default=1.0
        Angular scaling parameter.

    Returns
    -------
    float
        Characteristic effective Gaussian input NA.
    """
    u0 = gaussian_input_angular_width(beam_diameter_in, focal_length_in)
    return float(n_ang * np.sin(u0))


# ---------------------------------------------------------------------
# 6. Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example parameters
    beam_diameter_in = 6   # e.g. mm, 1/e^2 beam diameter at pupil/lens plane
    focal_length_in = 40.0    # same unit as beam_diameter_in, e.g. mm

    na_out = 0.14
    n_ang = 1.328             # or try 1.0 if you want an effective external-angle model

    f180 = 5.104           # GHz, example nominal backscattering shift
    gamma = 0.03             # GHz, example Lorentzian half-width

    # Exact mean shift ratio
    ratio_exact = mean_shift_ratio_exact_untruncated_gaussian_input(
        beam_diameter_in=beam_diameter_in,
        focal_length_in=focal_length_in,
        na_out=na_out,
        n_ang=n_ang,
    )

    print("=== Relative peak shift ===")
    print(f"Exact   <f_B>/f180 = {ratio_exact:.8f}")
    print(f"Exact relative shift = {ratio_exact - 1:.8e}")

    # Absolute shift
    result = na_shift_absolute_untruncated_gaussian_input(
        f180=f180,
        beam_diameter_in=beam_diameter_in,
        focal_length_in=focal_length_in,
        na_out=na_out,
        n_ang=n_ang,
    )

    print("\n=== Absolute peak shift ===")
    print(f"u0            = {result['u0_rad']:.6f} rad  ({result['u0_deg']:.3f} deg)")
    print(f"f180          = {result['f180']:.6f}")
    print(f"mean shift    = {result['mean_shift']:.6f}")
    print(f"delta_f_NA    = {result['delta_f_na']:.6f}")

    # Effective Gaussian input NA (optional diagnostic)
    na_eff = gaussian_input_effective_na(
        beam_diameter_in=beam_diameter_in,
        focal_length_in=focal_length_in,
        n_ang=n_ang,
    )
    print(f"NA_eff,input  = {na_eff:.6f}")

    # Optional full Lorentzian spectrum
    f_axis = np.linspace(4.9, 5.2, 400)
    spectrum = brillouin_spectrum_lorentzian_untruncated_gaussian_input(
        f_axis=f_axis,
        f180=f180,
        gamma=gamma,
        beam_diameter_in=beam_diameter_in,
        focal_length_in=focal_length_in,
        na_out=na_out,
        n_ang=n_ang,
    )

    print("\nComputed Lorentzian-broadened spectrum on f_axis.")