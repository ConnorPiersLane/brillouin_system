from __future__ import annotations
import numpy as np


def pupil_angle_limit(
    pupil_diameter: float,
    focal_length: float,
    n_sample: float = 1.328,
) -> float:
    theta_air = np.arctan((pupil_diameter / 2.0) / focal_length)
    return np.arcsin(np.sin(theta_air) / n_sample)


def collection_weight_uniform(v: np.ndarray) -> np.ndarray:
    # uniform power per solid angle
    return np.sin(v)


def brillouin_ratio_collection_only(v: np.ndarray) -> np.ndarray:
    # scattering angle = pi - v
    # f/f180 = sin((pi - v)/2) = cos(v/2)
    return np.cos(v / 2.0)


def mean_shift_ratio_collection_only_uniform(
    pupil_diameter_out: float,
    focal_length_out: float,
    n_sample: float = 1.328,
    n_v: int = 5000,
) -> float:
    alpha = pupil_angle_limit(pupil_diameter_out, focal_length_out, n_sample)

    v = np.linspace(0.0, alpha, n_v)

    W = collection_weight_uniform(v)
    ratio = brillouin_ratio_collection_only(v)

    numerator = np.trapezoid(W * ratio, x=v)
    denominator = np.trapezoid(W, x=v)

    return float(numerator / denominator)


def absolute_shift_collection_only_uniform(
    f180: float,
    pupil_diameter_out: float,
    focal_length_out: float,
    n_sample: float = 1.328,
) -> dict:
    ratio = mean_shift_ratio_collection_only_uniform(
        pupil_diameter_out=pupil_diameter_out,
        focal_length_out=focal_length_out,
        n_sample=n_sample,
    )

    alpha = pupil_angle_limit(
        pupil_diameter=pupil_diameter_out,
        focal_length=focal_length_out,
        n_sample=n_sample,
    )

    return {
        "mean_ratio": ratio,
        "relative_shift": ratio - 1.0,
        "f180": f180,
        "mean_shift": ratio * f180,
        "delta_f_NA": (ratio - 1.0) * f180,
        "alpha_out_deg": np.degrees(alpha),
    }


if __name__ == "__main__":
    # uniform collection
    f180 = 3.7  # GHz
    n_sample = 1.47 #1.328

    result_5x = absolute_shift_collection_only_uniform(
        f180=f180,
        pupil_diameter_out=11.2,
        focal_length_out=40.0,
        n_sample=n_sample,
    )

    result_20x = absolute_shift_collection_only_uniform(
        f180=f180,
        pupil_diameter_out=8.4,
        focal_length_out=10.0,
        n_sample=n_sample,
    )

    print("=== 5X uniform collection ===")
    for k, v in result_5x.items():
        print(f"{k}: {v}")

    print("\n=== 20X uniform collection ===")
    for k, v in result_20x.items():
        print(f"{k}: {v}")

    print("\nPredicted 20X - 5X frequency difference:")
    print(result_20x["mean_shift"] - result_5x["mean_shift"], "GHz")