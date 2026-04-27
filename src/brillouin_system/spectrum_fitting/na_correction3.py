from __future__ import annotations
import numpy as np


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


def gaussian_illumination_weight(angle: np.ndarray, angle0: float) -> np.ndarray:
    """
    Gaussian illumination angular weighting, including solid-angle factor.

    Approximation:
        W_i(u) = exp[-2 (u/u0)^2] sin(u)
    """
    return np.exp(-2.0 * (angle / angle0) ** 2) * np.sin(angle)


def uniform_collection_weight(angle: np.ndarray) -> np.ndarray:
    """
    Uniform collection over solid angle.

        W_c(v) = sin(v)
    """
    return np.sin(angle)


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

    f_B/f180 = |ks - ki| / 2k
    """
    arg = 0.5 * (
        1.0
        + np.cos(u) * np.cos(v)
        - np.sin(u) * np.sin(v) * np.cos(phi)
    )
    return np.sqrt(np.clip(arg, 0.0, None))


def mean_shift_ratio_gaussian_input_uniform_collection(
    beam_diameter_in: float,
    pupil_diameter: float,
    focal_length: float,
    n_sample: float = 1.328,
    n_u: int = 240,
    n_v: int = 240,
    n_phi: int = 180,
) -> float:
    """
    Mean Brillouin shift ratio <f_B>/f180.

    Illumination:
        Gaussian beam focused by the objective.

    Collection:
        Uniform over collected solid angle.

    Same objective is used for illumination and collection.
    """
    u0 = gaussian_angle_width(
        beam_diameter=beam_diameter_in,
        focal_length=focal_length,
        n_sample=n_sample,
    )

    alpha = pupil_angle_limit(
        pupil_diameter=pupil_diameter,
        focal_length=focal_length,
        n_sample=n_sample,
    )

    u = np.linspace(0.0, alpha, n_u)
    v = np.linspace(0.0, alpha, n_v)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    U = u[:, None, None]
    V = v[None, :, None]
    P = phi[None, None, :]

    Wi = gaussian_illumination_weight(U, u0)
    Wc = uniform_collection_weight(V)
    W = Wi * Wc

    ratio = brillouin_shift_ratio_exact(U, V, P)

    # Average over azimuth using simple periodic mean
    ratio_phi_avg = np.mean(ratio, axis=2)

    numerator = np.trapezoid(
        np.trapezoid(W[:, :, 0] * ratio_phi_avg, x=v, axis=1),
        x=u,
        axis=0,
    )

    denominator = np.trapezoid(
        np.trapezoid(W[:, :, 0], x=v, axis=1),
        x=u,
        axis=0,
    )

    return float(numerator / denominator)


def absolute_shift(
    f180: float,
    beam_diameter_in: float,
    pupil_diameter: float,
    focal_length: float,
    n_sample: float = 1.328,
) -> dict:
    ratio = mean_shift_ratio_gaussian_input_uniform_collection(
        beam_diameter_in=beam_diameter_in,
        pupil_diameter=pupil_diameter,
        focal_length=focal_length,
        n_sample=n_sample,
    )

    alpha = pupil_angle_limit(pupil_diameter, focal_length, n_sample)
    u0 = gaussian_angle_width(beam_diameter_in, focal_length, n_sample)

    return {
        "mean_ratio": ratio,
        "relative_shift": ratio - 1.0,
        "f180": f180,
        "mean_shift": ratio * f180,
        "delta_f_NA": (ratio - 1.0) * f180,
        "u0_deg": np.degrees(u0),
        "alpha_deg": np.degrees(alpha),
    }


if __name__ == "__main__":
    f180 = 5.055  # GHz
    n_sample = 1.328

    result_5x = absolute_shift(
        f180=f180,
        beam_diameter_in=6.0,
        pupil_diameter=11.2,
        focal_length=40.0,
        n_sample=n_sample,
    )

    result_20x = absolute_shift(
        f180=f180,
        beam_diameter_in=6.0,
        pupil_diameter=8.2,
        focal_length=10.0,
        n_sample=n_sample,
    )

    print("=== 5X ===")
    for k, v in result_5x.items():
        print(f"{k}: {v}")

    print("\n=== 20X ===")
    for k, v in result_20x.items():
        print(f"{k}: {v}")

    print("\nPredicted 20X - 5X frequency difference:")
    print(result_20x["mean_shift"] - result_5x["mean_shift"], "GHz")