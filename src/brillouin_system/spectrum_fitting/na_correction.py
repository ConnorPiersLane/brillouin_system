"""
Brillouin peak shift from finite illumination and collection NA
---------------------------------------------------------------

This script implements the model discussed in the conversation:

1) Exact geometric Brillouin shift for one ray pair:
       f_B(u, v, dphi) = f180 * sqrt((1 + cos(u)cos(v) + sin(u)sin(v)cos(dphi))/2)

2) Gaussian illumination weighting:
       W_i(u) = exp(-2 * (u / u0)^2) * sin(u)

3) Uniform collection weighting:
       W_c(v) = sin(v)

4) Mean detected Brillouin shift:
       <f_B> = weighted average over u, v, dphi

5) Small-angle approximation:
       <f_B>/f180 ≈ 1 - ( <u^2>_i + <v^2>_c ) / 8

6) Full Lorentzian-broadened spectrum:
       S(f) = integral of Lorentzians centered at f_B(u,v,dphi)

Author notes
------------
- na_in and na_out are external NAs.
- n_sample is the refractive index of the medium where Brillouin scattering occurs.
- Internal cone angles are computed as:
      alpha = arcsin(NA / n_sample)
- For a flat cuvette wall with a sample behind it, this is the right angle to use
  in the sample, assuming planar interfaces and neglecting aberration.

Units
-----
- Angles are in radians internally.
- Frequencies may be in any unit, as long as f180, gamma, and f_axis use the same unit.
"""

from __future__ import annotations

import numpy as np


def internal_half_angle(na: float, n_sample: float) -> float:
    """
    Convert an external numerical aperture into the internal half-angle inside the sample.

    Parameters
    ----------
    na : float
        External numerical aperture of illumination or collection.
    n_sample : float
        Refractive index of the sample where Brillouin scattering occurs.

    Returns
    -------
    float
        Internal half-angle in radians:
            alpha = arcsin(na / n_sample)

    Raises
    ------
    ValueError
        If na < 0, n_sample <= 0, or na / n_sample > 1.
    """
    if na < 0:
        raise ValueError("NA must be non-negative.")
    if n_sample <= 0:
        raise ValueError("n_sample must be positive.")
    ratio = na / n_sample
    if ratio > 1:
        raise ValueError(
            f"NA/n_sample = {ratio:.3f} > 1. This is not physically allowed."
        )
    return np.arcsin(ratio)


def gaussian_illumination_weight(u: np.ndarray, u0: float) -> np.ndarray:
    """
    Angular weighting of the incoming Gaussian beam.

    This models a Gaussian angular spectrum multiplied by the solid-angle Jacobian sin(u).

    W_i(u) = exp[-2 * (u/u0)^2] * sin(u)

    Parameters
    ----------
    u : np.ndarray
        Incoming angles in radians.
    u0 : float
        Characteristic angular width of the incoming Gaussian beam, in radians.

    Returns
    -------
    np.ndarray
        Illumination weights.
    """
    return np.exp(-2.0 * (u / u0) ** 2) * np.sin(u)


def uniform_collection_weight(v: np.ndarray) -> np.ndarray:
    """
    Angular weighting of the collected light.

    This follows the literature assumption of isotropic scattered radiation,
    giving a solid-angle weighting proportional to sin(v).

    Parameters
    ----------
    v : np.ndarray
        Collection angles in radians.

    Returns
    -------
    np.ndarray
        Collection weights W_c(v) = sin(v).
    """
    return np.sin(v)


def gaussian_collection_weight(v: np.ndarray, v0: float) -> np.ndarray:
    return np.exp(-2.0 * (v / v0) ** 2) * np.sin(v)

def effective_u0(alpha_in: float, fill_factor: float = 1.0) -> float:
    """
    Convert a dimensionless fill factor into the Gaussian angular width u0.

    We define:
        u0 = fill_factor * alpha_in

    Interpretation
    --------------
    fill_factor = 1.0
        The Gaussian beam roughly fills the incoming NA cone.
    fill_factor < 1.0
        The beam underfills the objective.
    fill_factor > 1.0
        The beam overfills/clips the pupil and approaches a more uniform filling.

    Parameters
    ----------
    alpha_in : float
        Internal illumination half-angle in radians.
    fill_factor : float, default=1.0
        Relative filling of the input pupil.

    Returns
    -------
    float
        Gaussian angular width u0 in radians.
    """
    if fill_factor <= 0:
        raise ValueError("fill_factor must be > 0.")
    return fill_factor * alpha_in


def brillouin_shift_ratio_exact(
    u: np.ndarray,
    v: np.ndarray,
    dphi: np.ndarray,
) -> np.ndarray:
    """
    Compute the exact ratio f_B / f180 for every ray-pair geometry.

    Geometry
    --------
    - u    : incoming polar angle from +z
    - v    : outgoing polar angle from -z
    - dphi : relative azimuth

    The scattering angle theta between the two rays satisfies:
        cos(theta) = -cos(u)cos(v) - sin(u)sin(v)cos(dphi)

    Then:
        f_B / f180 = sin(theta / 2)
                    = sqrt((1 - cos(theta))/2)
                    = sqrt((1 + cos(u)cos(v) + sin(u)sin(v)cos(dphi))/2)

    Parameters
    ----------
    u, v, dphi : np.ndarray
        Arrays broadcastable to the same shape.

    Returns
    -------
    np.ndarray
        Ratio f_B / f180.
    """
    arg = 0.5 * (
        1.0
        + np.cos(u) * np.cos(v)
        + np.sin(u) * np.sin(v) * np.cos(dphi)
    )

    # Small numerical guard against negative values due to floating-point noise.
    arg = np.clip(arg, 0.0, None)
    return np.sqrt(arg)


def mean_shift_ratio_exact(
    na_in: float,
    na_out: float,
    n_sample: float,
    fill_factor: float = 1.0,
    n_u: int = 160,
    n_v: int = 160,
    n_phi: int = 180,
) -> float:
    """
    Compute the exact weighted mean Brillouin shift ratio <f_B> / f180.

    This performs the 3D angular integral numerically:
        <f_B>/f180 = ∫∫∫ W_i(u) W_c(v) [f_B(u,v,dphi)/f180] du dv dphi
                     ---------------------------------------------------
                             ∫∫∫ W_i(u) W_c(v) du dv dphi

    Parameters
    ----------
    na_in : float
        Incoming NA.
    na_out : float
        Collection NA.
    n_sample : float
        Refractive index of the sample.
    fill_factor : float, default=1.0
        Controls Gaussian filling of the incoming pupil:
            u0 = fill_factor * alpha_in
    n_u, n_v, n_phi : int
        Number of grid points for u, v, and dphi integration.

    Returns
    -------
    float
        Mean shift ratio <f_B> / f180.
    """
    alpha_in = internal_half_angle(na_in, n_sample)
    alpha_out = internal_half_angle(na_out, n_sample)
    u0 = effective_u0(alpha_in, fill_factor)

    # Integration grids
    u = np.linspace(0.0, alpha_in, n_u)
    v = np.linspace(0.0, alpha_out, n_v)
    dphi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    # 3D broadcasted coordinate arrays
    U = u[:, None, None]
    V = v[None, :, None]
    P = dphi[None, None, :]

    # Angular weights
    Wi = gaussian_illumination_weight(U, u0)
    Wc = uniform_collection_weight(V)
    # Wc = gaussian_collection_weight(V, v0)
    W = Wi * Wc

    # Exact shift ratio
    ratio = brillouin_shift_ratio_exact(U, V, P)

    # Numerator and denominator of the weighted mean
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


def mean_u2_gaussian(
    na_in: float,
    n_sample: float,
    fill_factor: float = 1.0,
    n_u: int = 5000,
) -> float:
    """
    Compute <u^2> for the incoming Gaussian beam numerically.

    <u^2>_i = ∫ u^2 W_i(u) du / ∫ W_i(u) du

    Parameters
    ----------
    na_in : float
        Incoming NA.
    n_sample : float
        Refractive index of the sample.
    fill_factor : float, default=1.0
        Gaussian fill factor.
    n_u : int
        Number of integration points.

    Returns
    -------
    float
        Mean square incoming angle <u^2> in rad^2.
    """
    alpha_in = internal_half_angle(na_in, n_sample)
    u0 = effective_u0(alpha_in, fill_factor)

    u = np.linspace(0.0, alpha_in, n_u)
    Wi = gaussian_illumination_weight(u, u0)

    numerator = np.trapezoid((u**2) * Wi, u)
    denominator = np.trapezoid(Wi, u)
    return float(numerator / denominator)


def mean_v2_uniform(
    na_out: float,
    n_sample: float,
    n_v: int = 5000,
) -> float:
    """
    Compute <v^2> for the collection cone with uniform solid-angle weighting.

    <v^2>_c = ∫ v^2 sin(v) dv / ∫ sin(v) dv

    Parameters
    ----------
    na_out : float
        Collection NA.
    n_sample : float
        Refractive index of the sample.
    n_v : int
        Number of integration points.

    Returns
    -------
    float
        Mean square collection angle <v^2> in rad^2.
    """
    alpha_out = internal_half_angle(na_out, n_sample)

    v = np.linspace(0.0, alpha_out, n_v)
    Wc = uniform_collection_weight(v)

    numerator = np.trapezoid((v**2) * Wc, v)
    denominator = np.trapezoid(Wc, v)
    return float(numerator / denominator)


def mean_shift_ratio_small_angle(
    na_in: float,
    na_out: float,
    n_sample: float,
    fill_factor: float = 1.0,
) -> float:
    """
    Small-angle approximation for the mean Brillouin shift ratio.

    Formula
    -------
        <f_B>/f180 ≈ 1 - ( <u^2>_i + <v^2>_c ) / 8

    Parameters
    ----------
    na_in : float
        Incoming NA.
    na_out : float
        Collection NA.
    n_sample : float
        Refractive index of the sample.
    fill_factor : float, default=1.0
        Gaussian fill factor for the incoming beam.

    Returns
    -------
    float
        Approximate mean shift ratio <f_B> / f180.
    """
    u2 = mean_u2_gaussian(na_in, n_sample, fill_factor=fill_factor)
    v2 = mean_v2_uniform(na_out, n_sample)
    return 1.0 - (u2 + v2) / 8.0


def na_shift_absolute(
    f180: float,
    na_in: float,
    na_out: float,
    n_sample: float,
    fill_factor: float = 1.0,
    use_exact: bool = True,
) -> dict:
    """
    Compute the absolute NA-induced shift.

    Parameters
    ----------
    f180 : float
        The nominal 180-degree Brillouin shift in your preferred frequency unit
        (e.g. GHz).
    na_in : float
        Incoming NA.
    na_out : float
        Collection NA.
    n_sample : float
        Refractive index of the sample.
    fill_factor : float, default=1.0
        Gaussian fill factor for illumination.
    use_exact : bool, default=True
        If True, use the exact numerical angular average.
        If False, use the small-angle approximation.

    Returns
    -------
    dict
        Dictionary containing:
        - 'f180'
        - 'mean_ratio'
        - 'mean_shift'
        - 'delta_f_na'
    """
    if use_exact:
        ratio = mean_shift_ratio_exact(
            na_in=na_in,
            na_out=na_out,
            n_sample=n_sample,
            fill_factor=fill_factor,
        )
    else:
        ratio = mean_shift_ratio_small_angle(
            na_in=na_in,
            na_out=na_out,
            n_sample=n_sample,
            fill_factor=fill_factor,
        )

    mean_shift = ratio * f180
    delta_f_na = mean_shift - f180

    return {
        "f180": float(f180),
        "mean_ratio": float(ratio),
        "mean_shift": float(mean_shift),
        "delta_f_na": float(delta_f_na),
    }


def brillouin_spectrum_lorentzian(
    f_axis: np.ndarray,
    f180: float,
    gamma: float,
    na_in: float,
    na_out: float,
    n_sample: float,
    fill_factor: float = 1.0,
    n_u: int = 120,
    n_v: int = 120,
    n_phi: int = 120,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the full Lorentzian-broadened Brillouin spectrum:
        S(f) = ∫∫∫ W_i(u) W_c(v) * gamma / [(f - f_B)^2 + gamma^2] du dv dphi

    Parameters
    ----------
    f_axis : np.ndarray
        Frequency axis where the spectrum will be evaluated.
    f180 : float
        Nominal 180-degree Brillouin shift.
    gamma : float
        Lorentzian half-width (same units as f_axis and f180).
    na_in : float
        Incoming NA.
    na_out : float
        Collection NA.
    n_sample : float
        Refractive index of the sample.
    fill_factor : float, default=1.0
        Gaussian fill factor for the incoming beam.
    n_u, n_v, n_phi : int
        Integration grid sizes.
    normalize : bool, default=True
        If True, normalize the maximum of the spectrum to 1.

    Returns
    -------
    np.ndarray
        Spectrum evaluated on f_axis.
    """
    alpha_in = internal_half_angle(na_in, n_sample)
    alpha_out = internal_half_angle(na_out, n_sample)
    u0 = effective_u0(alpha_in, fill_factor)

    u = np.linspace(0.0, alpha_in, n_u)
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

    # Build the spectrum point-by-point.
    for i, f in enumerate(f_axis):
        lorentz = gamma / ((f - f_centers) ** 2 + gamma**2)
        val = np.trapezoid(
            np.trapezoid(
                np.trapezoid(W * lorentz, dphi, axis=2),
                v,
                axis=1,
            ),
            u,
            axis=0,
        )
        spectrum[i] = val

    if normalize and spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()

    return spectrum


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Example usage for your case:
    #   incoming NA  = 0.3
    #   collecting NA = 0.4
    #   sample is water, n ≈ 1.33
    # ------------------------------------------------------------
    na_in = 3/40
    na_out = 0.14
    na_in = 3/10
    na_out = 0.42
    n_sample = 1.328 #1.33 water

    # fill_factor = 1.0 means the incoming Gaussian roughly fills the input NA.
    # Try smaller values like 0.6 to simulate underfilling.
    fill_factor = 1.0

    # Exact mean shift ratio
    ratio_exact = mean_shift_ratio_exact(
        na_in=na_in,
        na_out=na_out,
        n_sample=n_sample,
        fill_factor=fill_factor,
    )

    # Small-angle approximation
    ratio_approx = mean_shift_ratio_small_angle(
        na_in=na_in,
        na_out=na_out,
        n_sample=n_sample,
        fill_factor=fill_factor,
    )

    print("=== Relative peak shift ===")
    print(f"Exact   <f_B>/f180 = {ratio_exact:.8f}")
    print(f"Approx  <f_B>/f180 = {ratio_approx:.8f}")
    print(f"Exact relative shift = {ratio_exact - 1:.8e}")
    print(f"Approx relative shift = {ratio_approx - 1:.8e}")

    # If you know the nominal backscattering Brillouin shift, give it here.
    # Example only:
    f180 = 5.104  # GHz, just as an example
    # f180 = 5.1

    result = na_shift_absolute(
        f180=f180,
        na_in=na_in,
        na_out=na_out,
        n_sample=n_sample,
        fill_factor=fill_factor,
        use_exact=True,
    )

    print("\n=== Absolute peak shift ===")
    print(f"f180       = {result['f180']:.6f} GHz")
    print(f"mean shift = {result['mean_shift']:.6f} GHz")
    print(f"delta_f_NA = {result['delta_f_na']:.6f} GHz")

    # # Optional: compute the full spectrum
    # f_axis = np.linspace(7.2, 7.7, 400)
    # gamma = 0.3  # GHz, example Lorentzian half-width
    #
    # spectrum = brillouin_spectrum_lorentzian(
    #     f_axis=f_axis,
    #     f180=f180,
    #     gamma=gamma,
    #     na_in=na_in,
    #     na_out=na_out,
    #     n_sample=n_sample,
    #     fill_factor=fill_factor,
    # )
    #
    # print("\nComputed a Lorentzian-broadened spectrum on f_axis.")