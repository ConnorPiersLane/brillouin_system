from __future__ import annotations
import numpy as np


def internal_half_angle_from_na(na: float, n_sample: float = 1.328) -> float:
    """
    Collection half-angle inside sample:
        alpha = arcsin(NA / n_sample)
    """
    if na < 0:
        raise ValueError("NA must be non-negative.")
    if n_sample <= 0:
        raise ValueError("n_sample must be positive.")
    if na / n_sample > 1:
        raise ValueError("NA / n_sample > 1 is not physical.")
    return np.arcsin(na / n_sample)


def brillouin_ratio_collection_only(v: np.ndarray) -> np.ndarray:
    """
    Collection-only backscatter model.

    Incoming beam is assumed axial:
        u = 0

    Scattered/collected ray has angle v from the backward optical axis.

    Then:
        f_B / f180 = cos(v/2)
    """
    return np.cos(v / 2.0)


def mean_shift_ratio_collection_only(
    na_out: float,
    n_sample: float = 1.328,
    n_v: int = 5000,
) -> float:
    """
    Mean Brillouin shift ratio <f_B>/f180 from collection NA only.

    Uniform collection over solid angle:
        W_c(v) = sin(v)

    Integral:
        <f_B>/f180 =
            ∫_0^alpha cos(v/2) sin(v) dv
            --------------------------------
            ∫_0^alpha sin(v) dv
    """
    alpha = internal_half_angle_from_na(na_out, n_sample)

    v = np.linspace(0.0, alpha, n_v)
    W = np.sin(v)
    ratio = brillouin_ratio_collection_only(v)

    numerator = np.trapezoid(ratio * W, x=v)
    denominator = np.trapezoid(W, x=v)

    return float(numerator / denominator)


def mean_shift_ratio_collection_only_analytic(
    na_out: float,
    n_sample: float = 1.328,
) -> float:
    """
    Analytic version of the collection-only model.

    Result:
        <f_B>/f180 =
        (2/3) * [1 - cos^3(alpha/2)] / [1 - cos^2(alpha/2)]
    """
    alpha = internal_half_angle_from_na(na_out, n_sample)
    c = np.cos(alpha / 2.0)

    return float((2.0 / 3.0) * (1.0 - c**3) / (1.0 - c**2))


def absolute_shift_collection_only(
    f180: float,
    na_out: float,
    n_sample: float = 1.328,
) -> dict:
    """
    Convert relative NA correction into GHz or whatever units f180 uses.
    """
    ratio = mean_shift_ratio_collection_only_analytic(
        na_out=na_out,
        n_sample=n_sample,
    )

    return {
        "na_out": float(na_out),
        "n_sample": float(n_sample),
        "alpha_deg": float(np.degrees(internal_half_angle_from_na(na_out, n_sample))),
        "mean_ratio": float(ratio),
        "relative_shift": float(ratio - 1.0),
        "f180": float(f180),
        "mean_shift": float(ratio * f180),
        "delta_f_NA": float((ratio - 1.0) * f180),
    }


if __name__ == "__main__":
    # only uniform collection
    f180 = 5.1   # GHz
    n_sample = 1.328

    # Examples
    cases = {
        "5X nominal NA 0.14": 0.14,
        "5X clipped NA 0.11": 0.11,
        "20X nominal NA 0.42": 0.41,
    }

    for name, na in cases.items():
        result = absolute_shift_collection_only(
            f180=f180,
            na_out=na,
            n_sample=n_sample,
        )

        print(f"\n=== {name} ===")
        for k, v in result.items():
            print(f"{k}: {v}")