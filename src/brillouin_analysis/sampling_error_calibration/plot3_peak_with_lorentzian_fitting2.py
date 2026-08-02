#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -----------------------------
# Models
# -----------------------------

def lorentzian(x, amp, x0, gamma, offset):
    """Continuous Lorentzian evaluated at x."""
    return offset + amp / (1.0 + ((x - x0) / gamma) ** 2)


def pixel_integrated_lorentzian(xpix, amp, x0, gamma, offset, pixel_width=1.0):
    """
    Average Lorentzian intensity over each pixel.

    Pixel centered at xpix integrates from xpix - 0.5 to xpix + 0.5.
    gamma is the Lorentzian HWHM in pixels.
    """
    a = xpix - pixel_width / 2
    b = xpix + pixel_width / 2

    integral = amp * gamma * (
        np.arctan((b - x0) / gamma) -
        np.arctan((a - x0) / gamma)
    )

    return offset + integral / pixel_width


def fit_one_phase(xpix, y, d_guess, gamma_guess):
    """Fit both the naive and pixel-integrated Lorentzian models."""
    amp_guess = np.max(y) - np.min(y)
    offset_guess = np.min(y)
    p0 = [amp_guess, d_guess, gamma_guess, offset_guess]

    bounds = (
        [0.0, np.min(xpix) - 1.0, 1e-6, -np.inf],
        [np.inf, np.max(xpix) + 1.0, np.inf, np.inf],
    )

    popt_naive, _ = curve_fit(
        lorentzian,
        xpix,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    popt_integrated, _ = curve_fit(
        pixel_integrated_lorentzian,
        xpix,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    return popt_naive, popt_integrated


# -----------------------------
# Simulation
# -----------------------------

def run_pixel_phase_fit_test(
    gamma_true=0.7,
    amp_true=1.0,
    offset_true=0.0,
    n_pixels=11,
    n_phase=201,
    d_example=0.25,
    noise_sigma=0.0,
    random_seed=1,
    x_halfwidth=6.0,
    dx=0.001,
):
    rng = np.random.default_rng(random_seed)

    if n_pixels % 2 == 0:
        raise ValueError("n_pixels must be odd.")

    half = n_pixels // 2
    xpix = np.arange(-half, half + 1, dtype=float)
    xfine = np.arange(-x_halfwidth, x_halfwidth + dx, dx)
    phases = np.linspace(0.0, 1.0, n_phase, endpoint=False)

    fitted_center_naive = []
    fitted_center_integrated = []
    popts_naive = []
    popts_integrated = []

    for d in phases:
        y = pixel_integrated_lorentzian(
            xpix,
            amp_true,
            d,
            gamma_true,
            offset_true,
        )

        if noise_sigma > 0:
            y = y + rng.normal(0.0, noise_sigma, size=y.shape)

        popt_naive, popt_integrated = fit_one_phase(xpix, y, d, gamma_true)

        popts_naive.append(popt_naive)
        popts_integrated.append(popt_integrated)
        fitted_center_naive.append(popt_naive[1])
        fitted_center_integrated.append(popt_integrated[1])

    popts_naive = np.array(popts_naive)
    popts_integrated = np.array(popts_integrated)
    fitted_center_naive = np.array(fitted_center_naive)
    fitted_center_integrated = np.array(fitted_center_integrated)

    bias_naive = fitted_center_naive - phases
    bias_integrated = fitted_center_integrated - phases

    # -----------------------------
    # Example phase for the two missing figures
    # -----------------------------

    y_example = pixel_integrated_lorentzian(
        xpix,
        amp_true,
        d_example,
        gamma_true,
        offset_true,
    )

    if noise_sigma > 0:
        # Use a deterministic separate example noise draw.
        y_example = y_example + rng.normal(0.0, noise_sigma, size=y_example.shape)

    popt_naive_ex, popt_integrated_ex = fit_one_phase(
        xpix,
        y_example,
        d_example,
        gamma_true,
    )

    true_continuous_ex = lorentzian(xfine, amp_true, d_example, gamma_true, offset_true)
    true_pixel_integrated_ex = pixel_integrated_lorentzian(
        xfine,
        amp_true,
        d_example,
        gamma_true,
        offset_true,
    )
    naive_fit_ex = lorentzian(xfine, *popt_naive_ex)
    integrated_fit_ex = pixel_integrated_lorentzian(xfine, *popt_integrated_ex)

    naive_fit_at_pixels_ex = lorentzian(xpix, *popt_naive_ex)
    integrated_fit_at_pixels_ex = pixel_integrated_lorentzian(xpix, *popt_integrated_ex)

    # -----------------------------
    # Plot: now 2 x 2, matching the earlier script structure
    # -----------------------------

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Center bias from Lorentzian fitting to pixel-integrated samples\n"
        f"gamma={gamma_true:.3f} px, n_pixels={n_pixels}, noise_sigma={noise_sigma:g}",
        fontsize=14,
    )

    # Missing figure 1: example walkthrough with true curve and fitted curves
    ax = axes[0, 0]
    ax.plot(xfine, true_continuous_ex, label="Original Lorentzian")
    ax.plot(xfine, true_pixel_integrated_ex, label="True pixel-integrated signal")
    ax.plot(xfine, naive_fit_ex, "--", label="Naive Lorentzian fit")
    ax.plot(xfine, integrated_fit_ex, ":", label="Pixel-integrated fit")
    ax.axvline(d_example, color="k", linestyle=":", label=f"true center = {d_example:.3f}")
    ax.axvline(popt_naive_ex[1], linestyle="--", label=f"naive center = {popt_naive_ex[1]:.6f}")
    ax.axvline(popt_integrated_ex[1], linestyle=":", label=f"integrated center = {popt_integrated_ex[1]:.6f}")
    ax.set_title("One example fit walkthrough")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Missing figure 2: sampled data and fitted model values at pixel centers
    ax = axes[0, 1]
    ax.plot(xfine, true_pixel_integrated_ex, alpha=0.7, label="True pixel-integrated signal")
    markerline, stemlines, baseline = ax.stem(
        xpix,
        y_example,
        linefmt="C1-",
        markerfmt="C1o",
        basefmt="k-",
        label="Pixel samples",
    )
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(markerline, markersize=6)
    ax.plot(xpix, naive_fit_at_pixels_ex, "x", markersize=7, label="Naive fit at pixels")
    ax.plot(xpix, integrated_fit_at_pixels_ex, "+", markersize=8, label="Integrated fit at pixels")
    ax.set_title(f"Discrete samples for d={d_example:.3f}")
    ax.set_xlabel("Pixel center")
    ax.set_ylabel("Sample value")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Recovered center over one-pixel walk
    ax = axes[1, 0]
    ax.plot(phases, fitted_center_naive, label="Naive Lorentzian fit")
    ax.plot(phases, fitted_center_integrated, "--", label="Pixel-integrated fit")
    ax.plot(phases, phases, "k:", label="True center")
    ax.set_xlabel("True sub-pixel phase")
    ax.set_ylabel("Fitted center")
    ax.set_title("Recovered center over one-pixel walk")
    ax.grid(True)
    ax.legend()

    # Bias over one-pixel walk
    ax = axes[1, 1]
    ax.plot(phases, bias_naive, label="Naive Lorentzian fit")
    ax.plot(phases, bias_integrated, "--", label="Pixel-integrated fit")
    ax.axhline(0, color="k", linestyle=":")
    ax.set_xlabel("True sub-pixel phase")
    ax.set_ylabel("Center bias [pixels]")
    ax.set_title("Fit bias vs pixel phase")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

    print(f"Example d = {d_example:.6f}")
    print(f"Naive fit center = {popt_naive_ex[1]:.12f}")
    print(f"Naive fit bias   = {popt_naive_ex[1] - d_example:.6e} pixels")
    print(f"Integrated fit center = {popt_integrated_ex[1]:.12f}")
    print(f"Integrated fit bias   = {popt_integrated_ex[1] - d_example:.6e} pixels")

    print("\nNaive Lorentzian fit:")
    print(f"  peak-to-peak bias = {np.ptp(bias_naive):.6e} pixels")
    print(f"  rms bias          = {np.sqrt(np.mean(bias_naive**2)):.6e} pixels")

    print("\nPixel-integrated fit:")
    print(f"  peak-to-peak bias = {np.ptp(bias_integrated):.6e} pixels")
    print(f"  rms bias          = {np.sqrt(np.mean(bias_integrated**2)):.6e} pixels")

    return {
        "phases": phases,
        "xpix": xpix,
        "xfine": xfine,
        "bias_naive": bias_naive,
        "bias_integrated": bias_integrated,
        "fitted_center_naive": fitted_center_naive,
        "fitted_center_integrated": fitted_center_integrated,
        "popts_naive": popts_naive,
        "popts_integrated": popts_integrated,
        "d_example": d_example,
        "y_example": y_example,
        "popt_naive_example": popt_naive_ex,
        "popt_integrated_example": popt_integrated_ex,
    }


if __name__ == "__main__":
    results = run_pixel_phase_fit_test(
        gamma_true=0.6,
        n_pixels=11,
        n_phase=201,
        d_example=0.25,
        noise_sigma=0.0,
    )
