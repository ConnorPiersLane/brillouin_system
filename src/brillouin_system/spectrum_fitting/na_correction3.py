from __future__ import annotations
import numpy as np


def internal_half_angle(na: float, n_ang: float) -> float:
    if na < 0:
        raise ValueError("NA must be non-negative.")
    if n_ang <= 0:
        raise ValueError("n_ang must be positive.")
    if na / n_ang > 1:
        raise ValueError("NA / n_ang > 1 is not physical.")
    return np.arcsin(na / n_ang)


def gaussian_angular_width(beam_diameter: float, focal_length: float) -> float:
    """
    beam_diameter: 1/e^2 Gaussian beam diameter at pupil/lens plane.
    focal_length: same length unit as beam_diameter.
    """
    if beam_diameter <= 0:
        raise ValueError("beam_diameter must be > 0.")
    if focal_length <= 0:
        raise ValueError("focal_length must be > 0.")
    return np.arctan((beam_diameter / 2.0) / focal_length)


def gaussian_weight(angle: np.ndarray, angle0: float) -> np.ndarray:
    """
    Gaussian angular weight including solid-angle factor.
    """
    return np.exp(-2.0 * (angle / angle0) ** 2) * np.sin(angle)


def brillouin_shift_ratio_exact(
    u: np.ndarray,
    v: np.ndarray,
    dphi: np.ndarray,
) -> np.ndarray:
    """
    f_B / f180 for exact ray-pair geometry.
    """
    arg = 0.5 * (
        1.0
        + np.cos(u) * np.cos(v)
        + np.sin(u) * np.sin(v) * np.cos(dphi)
    )
    return np.sqrt(np.clip(arg, 0.0, None))


def mean_shift_ratio_gaussian_in_out(
    na_in: float,
    na_out: float,
    n_ang: float,
    beam_diameter_in: float,
    focal_length_in: float,
    beam_diameter_out: float,
    focal_length_out: float,
    n_u: int = 220,
    n_v: int = 220,
    n_phi: int = 180,
) -> float:
    """
    Exact mean Brillouin shift ratio <f_B>/f180.

    Gaussian input and Gaussian output, both truncated by NA.
    """

    alpha_in = internal_half_angle(na_in, n_ang)
    alpha_out = internal_half_angle(na_out, n_ang)

    u0 = gaussian_angular_width(beam_diameter_in, focal_length_in)
    v0 = gaussian_angular_width(beam_diameter_out, focal_length_out)

    u = np.linspace(0.0, alpha_in, n_u)
    v = np.linspace(0.0, alpha_out, n_v)
    dphi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    U = u[:, None, None]
    V = v[None, :, None]
    P = dphi[None, None, :]

    Wi = gaussian_weight(U, u0)
    Wc = gaussian_weight(V, v0)
    W = Wi * Wc

    ratio = brillouin_shift_ratio_exact(U, V, P)

    numerator = np.trapezoid(
        np.trapezoid(
            np.trapezoid(W * ratio, x=dphi, axis=2),
            x=v,
            axis=1,
        ),
        x=u,
        axis=0,
    )

    denominator = (2.0 * np.pi) * np.trapezoid(
        np.trapezoid(W[..., 0], x=v, axis=1),
        x=u,
        axis=0,
    )

    return float(numerator / denominator)


def na_shift_absolute_gaussian_in_out(
    f180: float,
    na_in: float,
    na_out: float,
    n_ang: float,
    beam_diameter_in: float,
    focal_length_in: float,
    beam_diameter_out: float,
    focal_length_out: float,
) -> dict:
    ratio = mean_shift_ratio_gaussian_in_out(
        na_in=na_in,
        na_out=na_out,
        n_ang=n_ang,
        beam_diameter_in=beam_diameter_in,
        focal_length_in=focal_length_in,
        beam_diameter_out=beam_diameter_out,
        focal_length_out=focal_length_out,
    )

    return {
        "mean_ratio": ratio,
        "f180": f180,
        "mean_shift": ratio * f180,
        "delta_f_NA": ratio * f180 - f180,
        "u0_rad": gaussian_angular_width(beam_diameter_in, focal_length_in),
        "v0_rad": gaussian_angular_width(beam_diameter_out, focal_length_out),
        "u0_deg": np.degrees(gaussian_angular_width(beam_diameter_in, focal_length_in)),
        "v0_deg": np.degrees(gaussian_angular_width(beam_diameter_out, focal_length_out)),
    }


def brillouin_spectrum_lorentzian_gaussian_in_out(
    f_axis: np.ndarray,
    f180: float,
    gamma: float,
    na_in: float,
    na_out: float,
    n_ang: float,
    beam_diameter_in: float,
    focal_length_in: float,
    beam_diameter_out: float,
    focal_length_out: float,
    n_u: int = 160,
    n_v: int = 160,
    n_phi: int = 120,
    normalize: bool = True,
) -> np.ndarray:
    """
    Lorentzian spectrum with Gaussian input and Gaussian output,
    both truncated by NA.
    """

    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    alpha_in = internal_half_angle(na_in, n_ang)
    alpha_out = internal_half_angle(na_out, n_ang)

    u0 = gaussian_angular_width(beam_diameter_in, focal_length_in)
    v0 = gaussian_angular_width(beam_diameter_out, focal_length_out)

    u = np.linspace(0.0, alpha_in, n_u)
    v = np.linspace(0.0, alpha_out, n_v)
    dphi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    U = u[:, None, None]
    V = v[None, :, None]
    P = dphi[None, None, :]

    Wi = gaussian_weight(U, u0)
    Wc = gaussian_weight(V, v0)
    W = Wi * Wc

    ratio = brillouin_shift_ratio_exact(U, V, P)
    f_centers = f180 * ratio

    spectrum = np.zeros_like(f_axis, dtype=float)

    for i, f in enumerate(f_axis):
        lorentz = gamma / ((f - f_centers) ** 2 + gamma**2)
        tmp_phi = np.trapezoid(W * lorentz, x=dphi, axis=2)
        spectrum[i] = np.trapezoid(
            np.trapezoid(tmp_phi, x=v, axis=1),
            x=u,
            axis=0,
        )

    if normalize and spectrum.max() > 0:
        spectrum /= spectrum.max()

    return spectrum


if __name__ == "__main__":
    na_in = 0.30
    na_out = 0.42
    n_ang = 1.328

    beam_diameter_in = 6.0
    focal_length_in = 10.0

    beam_diameter_out = 8.4
    focal_length_out = 10.0

    f180 = 5.104
    gamma = 0.03

    result = na_shift_absolute_gaussian_in_out(
        f180=f180,
        na_in=na_in,
        na_out=na_out,
        n_ang=n_ang,
        beam_diameter_in=beam_diameter_in,
        focal_length_in=focal_length_in,
        beam_diameter_out=beam_diameter_out,
        focal_length_out=focal_length_out,
    )

    print("=== Gaussian input + Gaussian output, NA cutoff ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    f_axis = np.linspace(4.9, 5.2, 400)
    spectrum = brillouin_spectrum_lorentzian_gaussian_in_out(
        f_axis=f_axis,
        f180=f180,
        gamma=gamma,
        na_in=na_in,
        na_out=na_out,
        n_ang=n_ang,
        beam_diameter_in=beam_diameter_in,
        focal_length_in=focal_length_in,
        beam_diameter_out=beam_diameter_out,
        focal_length_out=focal_length_out,
    )