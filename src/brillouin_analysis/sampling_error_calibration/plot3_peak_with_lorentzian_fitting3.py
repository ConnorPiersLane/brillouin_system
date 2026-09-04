#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf


# -----------------------------
# Models
# -----------------------------

def lorentzian(x, amp, x0, gamma, offset):
    """Continuous Lorentzian evaluated at x."""
    return offset + amp / (1.0 + ((x - x0) / gamma) ** 2)


def pixel_integrated_lorentzian(xpix, amp, x0, gamma, offset, pixel_width=1.0):
    """
    Average Lorentzian intensity over a perfect top-hat pixel.
    """
    a = xpix - pixel_width / 2
    b = xpix + pixel_width / 2

    integral = amp * gamma * (
            np.arctan((b - x0) / gamma) -
            np.arctan((a - x0) / gamma)
    )
    return offset + integral / pixel_width


def gaussian_blurred_pixel_integrated_lorentzian(xpix, amp, x0, gamma, offset, pixel_width=1.0, sigma_diff=0.25):
    """
    Average Lorentzian intensity over a soft-edged pixel blurred by charge diffusion.
    Uses numerical integration across a dense grid for each pixel element.
    """
    # Create an array output to match xpix shape
    xpix_arr = np.atleast_1d(xpix)
    out = np.zeros_like(xpix_arr, dtype=float)

    # Grid parameters per pixel for numerical integration
    n_sub = 100

    for i, xp in enumerate(xpix_arr):
        a = xp - pixel_width / 2
        b = xp + pixel_width / 2

        # Integrate out to 3-sigma past the physical pixel boundaries
        x_min = a - 3.0 * sigma_diff
        x_max = b + 3.0 * sigma_diff
        xs = np.linspace(x_min, x_max, n_sub)
        dx = xs[1] - xs[0]

        # Calculate the underlying continuous Lorentzian profile
        lorentz_vals = amp / (1.0 + ((xs - x0) / gamma) ** 2)

        # Analytical rect convolved with Gaussian is a difference of erfs
        window = 0.5 * (erf((xs - a) / (np.sqrt(2) * sigma_diff)) -
                        erf((xs - b) / (np.sqrt(2) * sigma_diff)))

        # Normalize response window to preserve total charge energy
        window_sum = np.sum(window) * dx
        if window_sum > 0:
            window /= window_sum

        out[i] = offset + np.sum(lorentz_vals * window) * dx

    return out if np.ndim(xpix) > 0 else out[0]


def fit_one_phase(xpix, y, d_guess, gamma_guess):
    """Fit naive, top-hat integrated, and diffusion-corrected models."""
    amp_guess = np.max(y) - np.min(y)
    offset_guess = np.min(y)
    p0 = [amp_guess, d_guess, gamma_guess, offset_guess]

    bounds = (
        [0.0, np.min(xpix) - 1.0, 1e-6, -np.inf],
        [np.inf, np.max(xpix) + 1.0, np.inf, np.inf],
    )

    # 1. Naive continuous Lorentzian model
    popt_naive, _ = curve_fit(lorentzian, xpix, y, p0=p0, bounds=bounds, maxfev=10000)

    # 2. Perfect Top-Hat Pixel Integrated model (Your old fix)
    popt_integrated, _ = curve_fit(pixel_integrated_lorentzian, xpix, y, p0=p0, bounds=bounds, maxfev=10000)

    # 3. Diffusion-Corrected Model (Fitting amp, x0, gamma, offset; fixing sigma_diff)
    # Note: You can also leave sigma_diff as a free parameter if fitting real data!
    def fit_diffusion_model(x, a, x0, g, off):
        return gaussian_blurred_pixel_integrated_lorentzian(x, a, x0, g, off, sigma_diff=0.25)

    popt_diffusion, _ = curve_fit(fit_diffusion_model, xpix, y, p0=p0, bounds=bounds, maxfev=10000)

    return popt_naive, popt_integrated, popt_diffusion


# -----------------------------
# Simulation
# -----------------------------

def run_pixel_phase_fit_test(
        gamma_true=0.6,
        amp_true=1.0,
        offset_true=0.0,
        sigma_diff_true=0.25,  # Realistic EMCCD diffusion width in pixels
        n_pixels=11,
        n_phase=201,
        d_example=0.25,
        noise_sigma=0.0,
        random_seed=1,
        x_halfwidth=6.0,
        dx=0.01,
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
    fitted_center_diffusion = []

    print("Running phase walk simulation...")
    for d in phases:
        # Generate data containing the real-world charge diffusion blurring artifact
        y = gaussian_blurred_pixel_integrated_lorentzian(
            xpix, amp_true, d, gamma_true, offset_true, sigma_diff=sigma_diff_true
        )

        if noise_sigma > 0:
            y = y + rng.normal(0.0, noise_sigma, size=y.shape)

        popt_naive, popt_integrated, popt_diffusion = fit_one_phase(xpix, y, d, gamma_true)

        fitted_center_naive.append(popt_naive[1])
        fitted_center_integrated.append(popt_integrated[1])
        fitted_center_diffusion.append(popt_diffusion[1])

    fitted_center_naive = np.array(fitted_center_naive)
    fitted_center_integrated = np.array(fitted_center_integrated)
    fitted_center_diffusion = np.array(fitted_center_diffusion)

    bias_naive = fitted_center_naive - phases
    bias_integrated = fitted_center_integrated - phases
    bias_diffusion = fitted_center_diffusion - phases

    # -----------------------------
    # Example calculations for plots
    # -----------------------------
    y_example = gaussian_blurred_pixel_integrated_lorentzian(
        xpix, amp_true, d_example, gamma_true, offset_true, sigma_diff=sigma_diff_true
    )

    popt_naive_ex, popt_integrated_ex, popt_diffusion_ex = fit_one_phase(
        xpix, y_example, d_example, gamma_true
    )

    true_continuous_ex = lorentzian(xfine, amp_true, d_example, gamma_true, offset_true)
    true_pixel_blur_ex = gaussian_blurred_pixel_integrated_lorentzian(
        xfine, amp_true, d_example, gamma_true, offset_true, sigma_diff=sigma_diff_true
    )

    naive_fit_ex = lorentzian(xfine, *popt_naive_ex)
    integrated_fit_ex = pixel_integrated_lorentzian(xfine, *popt_integrated_ex)
    diffusion_fit_ex = gaussian_blurred_pixel_integrated_lorentzian(
        xfine, *popt_diffusion_ex, sigma_diff=sigma_diff_true
    )

    # -----------------------------
    # Plotting
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Lorentzian Fitting Residuals with Charge Diffusion Blur\n"
        f"gamma={gamma_true:.2f} px, true $\sigma_{{diff}}$={sigma_diff_true:.2f} px, noise={noise_sigma:g}",
        fontsize=14,
    )

    # Walkthrough
    ax = axes[0, 0]
    ax.plot(xfine, true_continuous_ex, label="Original Profile")
    ax.plot(xfine, true_pixel_blur_ex, label="True Diffused Profile")
    ax.plot(xfine, naive_fit_ex, "--", label="Naive Fit")
    ax.plot(xfine, integrated_fit_ex, "-.", label="Top-Hat Integrated Fit")
    ax.plot(xfine, diffusion_fit_ex, ":", label="Diffusion-Corrected Fit", linewidth=2)
    ax.set_title("Profile Shapes")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Discrete sample visualization
    ax = axes[0, 1]
    ax.stem(xpix, y_example, linefmt="C1-", markerfmt="C1o", basefmt="k-", label="Pixel Data (with diffusion)")
    ax.plot(xpix, lorentzian(xpix, *popt_naive_ex), "x", label="Naive model at pixels")
    ax.plot(xpix, pixel_integrated_lorentzian(xpix, *popt_integrated_ex), "+", label="Top-Hat model at pixels")
    ax.plot(xpix, gaussian_blurred_pixel_integrated_lorentzian(xpix, *popt_diffusion_ex, sigma_diff=sigma_diff_true),
            "s", fillstyle='none', label="Diffusion model at pixels")
    ax.set_title(f"Discrete Samples (Phase = {d_example})")
    ax.set_xlabel("Pixel Center")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Recovered Center
    ax = axes[1, 0]
    ax.plot(phases, fitted_center_naive, label="Naive")
    ax.plot(phases, fitted_center_integrated, "--", label="Top-Hat Integrated")
    ax.plot(phases, fitted_center_diffusion, ":", label="Diffusion-Corrected", linewidth=2.5)
    ax.plot(phases, phases, "k:", alpha=0.5, label="True Line")
    ax.set_xlabel("True sub-pixel phase")
    ax.set_ylabel("Fitted center")
    ax.set_title("Recovered center over one-pixel walk")
    ax.grid(True)
    ax.legend()

    # Residual Bias Curve
    ax = axes[1, 1]
    ax.plot(phases, bias_naive, label="Naive")
    ax.plot(phases, bias_integrated, "--", label="Top-Hat Integrated (Shows physical mismatch error)")
    ax.plot(phases, bias_diffusion, ":", label="Diffusion-Corrected", linewidth=2.5)
    ax.axhline(0, color="k", linestyle=":")
    ax.set_xlabel("True sub-pixel phase")
    ax.set_ylabel("Center bias [pixels]")
    ax.set_title("Fit bias vs pixel phase")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Metrics
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Naive Fit Peak-to-Peak Bias:      {np.ptp(bias_naive):.6e} px")
    print(
        f"Top-Hat Fit Peak-to-Peak Bias:    {np.ptp(bias_integrated):.6e} px  <-- (The residual wave you see in Image 1!)")
    print(f"Diffusion Fit Peak-to-Peak Bias:  {np.ptp(bias_diffusion):.6e} px  <-- (Eliminated down to numeric noise)")

    return {"phases": phases, "bias_naive": bias_naive, "bias_integrated": bias_integrated,
            "bias_diffusion": bias_diffusion}


if __name__ == "__main__":
    results = run_pixel_phase_fit_test(
        gamma_true=0.6,
        sigma_diff_true=0.25,
        n_pixels=11,
        n_phase=101,
        d_example=0.25,
        noise_sigma=0.0
    )