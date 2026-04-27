from __future__ import annotations
import numpy as np


def angle_from_radius(radius: float, focal_length: float) -> float:
    """Exact ray angle from pupil radius and focal length."""
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if focal_length <= 0:
        raise ValueError("focal_length must be > 0.")
    return np.arctan(radius / focal_length)


def gaussian_angle_width(
    beam_diameter: float,
    focal_length: float,
    n_sample: float = 1.328,
) -> float:
    theta_air = np.arctan((beam_diameter / 2.0) / focal_length)
    return np.arcsin(np.sin(theta_air) / n_sample)


def pupil_angle_limit(
    pupil_diameter: float,
    focal_length: float,
    n_sample: float = 1.328,
) -> float:
    theta_air = np.arctan((pupil_diameter / 2.0) / focal_length)
    return np.arcsin(np.sin(theta_air) / n_sample)


def gaussian_weight(angle: np.ndarray, angle0: float) -> np.ndarray:
    """
    Gaussian angular weighting including solid-angle factor.

    W(a) = exp[-2(a/a0)^2] sin(a)
    """
    return np.exp(-2.0 * (angle / angle0) ** 2) * np.sin(angle)


def brillouin_shift_ratio_exact(
    u: np.ndarray,
    v: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Exact f_B / f180 for backscattering ray-pair geometry.

    u   = incoming polar angle from +z
    v   = collected polar angle from -z
    phi = relative azimuth

    f_B/f180 = sqrt((1 + cos(u)cos(v) + sin(u)sin(v)cos(phi))/2)
    """
    arg = 0.5 * (
        1.0
        + np.cos(u) * np.cos(v)
        - np.sin(u) * np.sin(v) * np.cos(phi)
    )
    return np.sqrt(np.clip(arg, 0.0, None))


def mean_shift_ratio_gaussian_pupil_limited(
    beam_diameter_in: float,
    beam_diameter_out: float,
    pupil_diameter_in: float,
    pupil_diameter_out: float,
    focal_length_in: float,
    focal_length_out: float,
    n_u: int = 240,
    n_v: int = 240,
    n_phi: int = 180,
) -> float:
    """
    Mean Brillouin shift ratio <f_B>/f180.

    Input:
        Gaussian illumination, integrated to input pupil limit.

    Output:
        Gaussian collection acceptance, integrated to output pupil limit.

    No small-angle approximation.
    """

    u0 = gaussian_angle_width(beam_diameter_in, focal_length_in)
    v0 = gaussian_angle_width(beam_diameter_out, focal_length_out)

    alpha_in = pupil_angle_limit(pupil_diameter_in, focal_length_in)
    alpha_out = pupil_angle_limit(pupil_diameter_out, focal_length_out)

    u = np.linspace(0.0, alpha_in, n_u)
    v = np.linspace(0.0, alpha_out, n_v)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    U = u[:, None, None]
    V = v[None, :, None]
    P = phi[None, None, :]

    Wi = gaussian_weight(U, u0)
    Wc = gaussian_weight(V, v0)
    W = Wi * Wc

    ratio = brillouin_shift_ratio_exact(U, V, P)

    numerator = np.trapezoid(
        np.trapezoid(
            np.trapezoid(W * ratio, x=phi, axis=2),
            x=v,
            axis=1,
        ),
        x=u,
        axis=0,
    )

    denominator = 2.0 * np.pi * np.trapezoid(
        np.trapezoid(W[..., 0], x=v, axis=1),
        x=u,
        axis=0,
    )

    return float(numerator / denominator)


def absolute_shift_gaussian_pupil_limited(
    f180: float,
    beam_diameter_in: float,
    beam_diameter_out: float,
    pupil_diameter_in: float,
    pupil_diameter_out: float,
    focal_length_in: float,
    focal_length_out: float,
) -> dict:
    ratio = mean_shift_ratio_gaussian_pupil_limited(
        beam_diameter_in=beam_diameter_in,
        beam_diameter_out=beam_diameter_out,
        pupil_diameter_in=pupil_diameter_in,
        pupil_diameter_out=pupil_diameter_out,
        focal_length_in=focal_length_in,
        focal_length_out=focal_length_out,
    )

    u0 = gaussian_angle_width(beam_diameter_in, focal_length_in)
    v0 = gaussian_angle_width(beam_diameter_out, focal_length_out)
    alpha_in = pupil_angle_limit(pupil_diameter_in, focal_length_in)
    alpha_out = pupil_angle_limit(pupil_diameter_out, focal_length_out)

    return {
        "mean_ratio": ratio,
        "relative_shift": ratio - 1.0,
        "f180": f180,
        "mean_shift": ratio * f180,
        "delta_f_NA": (ratio - 1.0) * f180,
        "u0_deg": np.degrees(u0),
        "v0_deg": np.degrees(v0),
        "alpha_in_deg": np.degrees(alpha_in),
        "alpha_out_deg": np.degrees(alpha_out),
    }


def spectrum_lorentzian_gaussian_pupil_limited(
    f_axis: np.ndarray,
    f180: float,
    gamma: float,
    beam_diameter_in: float,
    beam_diameter_out: float,
    pupil_diameter_in: float,
    pupil_diameter_out: float,
    focal_length_in: float,
    focal_length_out: float,
    n_u: int = 160,
    n_v: int = 160,
    n_phi: int = 120,
    normalize: bool = True,
) -> np.ndarray:
    u0 = gaussian_angle_width(beam_diameter_in, focal_length_in)
    v0 = gaussian_angle_width(beam_diameter_out, focal_length_out)

    alpha_in = pupil_angle_limit(pupil_diameter_in, focal_length_in)
    alpha_out = pupil_angle_limit(pupil_diameter_out, focal_length_out)

    u = np.linspace(0.0, alpha_in, n_u)
    v = np.linspace(0.0, alpha_out, n_v)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    U = u[:, None, None]
    V = v[None, :, None]
    P = phi[None, None, :]

    W = gaussian_weight(U, u0) * gaussian_weight(V, v0)
    centers = f180 * brillouin_shift_ratio_exact(U, V, P)

    spectrum = np.zeros_like(f_axis, dtype=float)

    for i, f in enumerate(f_axis):
        L = gamma / ((f - centers) ** 2 + gamma**2)
        tmp = np.trapezoid(W * L, x=phi, axis=2)
        spectrum[i] = np.trapezoid(
            np.trapezoid(tmp, x=v, axis=1),
            x=u,
            axis=0,
        )

    if normalize and spectrum.max() > 0:
        spectrum /= spectrum.max()

    return spectrum


if __name__ == "__main__":
    f180 = 5.10  # GHz

    # Example 5X
    result_5x = absolute_shift_gaussian_pupil_limited(
        f180=f180,
        beam_diameter_in=6,       # mm, your incoming 1/e^2 beam diameter
        beam_diameter_out=6,      # mm, F810 output 1/e^2 beam diameter
        pupil_diameter_in=11.2,     # mm, 5X pupil diameter
        pupil_diameter_out=9,    # mm, 5X pupil diameter
        focal_length_in=40.0,       # mm, 5X focal length
        focal_length_out=40.0,      # mm, 5X focal length
    )

    print("=== 5X ===")
    for k, v in result_5x.items():
        print(f"{k}: {v}")

    # Example 20X
    result_20x = absolute_shift_gaussian_pupil_limited(
        f180=f180,
        beam_diameter_in=6,       # mm, incoming 1/e^2 beam diameter
        beam_diameter_out=6,      # mm, F810 output 1/e^2 beam diameter
        pupil_diameter_in=8.4,      # mm, 20X pupil diameter
        pupil_diameter_out=8.4,     # mm, 20X pupil diameter
        focal_length_in=10.0,       # mm, 20X focal length
        focal_length_out=10.0,      # mm, 20X focal length
    )

    print("\n=== 20X ===")
    for k, v in result_20x.items():
        print(f"{k}: {v}")

    print("\nPredicted 20X - 5X frequency difference:")
    print(result_20x["mean_shift"] - result_5x["mean_shift"], "GHz")