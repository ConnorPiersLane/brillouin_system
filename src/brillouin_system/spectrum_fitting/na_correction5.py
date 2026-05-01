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


def gaussian_collection_weight(v: np.ndarray, v0: float) -> np.ndarray:
    return np.exp(-2.0 * (v / v0) ** 2) * np.sin(v)


def brillouin_ratio_collection_only(v: np.ndarray) -> np.ndarray:
    return np.cos(v / 2.0)


def mean_shift_ratio_collection_only_gaussian(
    beam_diameter_out: float,
    pupil_diameter_out: float,
    focal_length_out: float,
    n_sample: float = 1.328,
    n_v: int = 5000,
) -> float:
    v0 = gaussian_angle_width(beam_diameter_out, focal_length_out, n_sample)
    alpha = pupil_angle_limit(pupil_diameter_out, focal_length_out, n_sample)

    v = np.linspace(0.0, alpha, n_v)

    W = gaussian_collection_weight(v, v0)
    ratio = brillouin_ratio_collection_only(v)

    numerator = np.trapezoid(W * ratio, x=v)
    denominator = np.trapezoid(W, x=v)

    return float(numerator / denominator)


def absolute_shift_collection_only_gaussian(
    f180: float,
    beam_diameter_out: float,
    pupil_diameter_out: float,
    focal_length_out: float,
    n_sample: float = 1.328,
) -> dict:
    ratio = mean_shift_ratio_collection_only_gaussian(
        beam_diameter_out=beam_diameter_out,
        pupil_diameter_out=pupil_diameter_out,
        focal_length_out=focal_length_out,
        n_sample=n_sample,
    )

    v0 = gaussian_angle_width(beam_diameter_out, focal_length_out, n_sample)
    alpha = pupil_angle_limit(pupil_diameter_out, focal_length_out, n_sample)

    return {
        "mean_ratio": ratio,
        "relative_shift": ratio - 1.0,
        "f180": f180,
        "mean_shift": ratio * f180,
        "delta_f_NA": (ratio - 1.0) * f180,
        "v0_deg": np.degrees(v0),
        "alpha_out_deg": np.degrees(alpha),
    }


if __name__ == "__main__":
    # collection only but gaussian
    f180 = 5.065  # GHz
    n_sample = 1.328

    result_5x = absolute_shift_collection_only_gaussian(
        f180=f180,
        beam_diameter_out=7.5,      # mm, Gaussian collection mode at pupil
        pupil_diameter_out=9.0,     # mm, clipped collection aperture mapped to pupil
        focal_length_out=40.0,      # mm
        n_sample=n_sample,
    )

    result_20x = absolute_shift_collection_only_gaussian(
        f180=f180,
        beam_diameter_out=7.5,      # mm, Gaussian collection mode at pupil
        pupil_diameter_out=8.4,     # mm, objective pupil / collection aperture
        focal_length_out=10.0,      # mm
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