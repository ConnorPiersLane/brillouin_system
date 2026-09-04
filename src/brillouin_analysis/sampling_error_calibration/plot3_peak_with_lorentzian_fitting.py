import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -----------------------------
# Models
# -----------------------------

def lorentzian(x, amp, x0, gamma, offset):
    return offset + amp / (1.0 + ((x - x0) / gamma) ** 2)


def pixel_integrated_lorentzian(xpix, amp, x0, gamma, offset, pixel_width=1.0):
    """
    Average Lorentzian intensity over each pixel.

    Pixel centered at xpix integrates from xpix - 0.5 to xpix + 0.5.
    """
    a = xpix - pixel_width / 2
    b = xpix + pixel_width / 2

    integral = amp * gamma * (
        np.arctan((b - x0) / gamma) -
        np.arctan((a - x0) / gamma)
    )

    return offset + integral / pixel_width


# -----------------------------
# Simulation
# -----------------------------

def run_pixel_phase_fit_test(
    gamma_true=0.7,
    amp_true=1.0,
    offset_true=0.0,
    n_pixels=11,
    n_phase=201,
    noise_sigma=0.0,
    random_seed=1,
):
    rng = np.random.default_rng(random_seed)

    if n_pixels % 2 == 0:
        raise ValueError("n_pixels must be odd.")

    half = n_pixels // 2
    xpix = np.arange(-half, half + 1, dtype=float)

    phases = np.linspace(0.0, 1.0, n_phase, endpoint=False)

    fitted_center_naive = []
    fitted_center_integrated = []

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

        p0 = [np.max(y) - np.min(y), d, gamma_true, np.min(y)]

        # This is the "wrong" fit:
        # fit a continuous Lorentzian to pixel-integrated data.
        popt_naive, _ = curve_fit(
            lorentzian,
            xpix,
            y,
            p0=p0,
            maxfev=10000,
        )

        # This is the correct comparison:
        # fit the actual pixel-integrated model.
        popt_integrated, _ = curve_fit(
            pixel_integrated_lorentzian,
            xpix,
            y,
            p0=p0,
            maxfev=10000,
        )

        fitted_center_naive.append(popt_naive[1])
        fitted_center_integrated.append(popt_integrated[1])

    fitted_center_naive = np.array(fitted_center_naive)
    fitted_center_integrated = np.array(fitted_center_integrated)

    bias_naive = fitted_center_naive - phases
    bias_integrated = fitted_center_integrated - phases

    # -----------------------------
    # Plot
    # -----------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(phases, fitted_center_naive, label="Naive Lorentzian fit")
    axes[0].plot(phases, fitted_center_integrated, "--", label="Pixel-integrated fit")
    axes[0].plot(phases, phases, "k:", label="True center")
    axes[0].set_xlabel("True sub-pixel phase")
    axes[0].set_ylabel("Fitted center")
    axes[0].set_title("Recovered center")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(phases, bias_naive, label="Naive Lorentzian fit")
    axes[1].plot(phases, bias_integrated, "--", label="Pixel-integrated fit")
    axes[1].axhline(0, color="k", linestyle=":")
    axes[1].set_xlabel("True sub-pixel phase")
    axes[1].set_ylabel("Center bias [pixels]")
    axes[1].set_title("Fit bias vs pixel phase")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("Naive Lorentzian fit:")
    print(f"  peak-to-peak bias = {np.ptp(bias_naive):.6e} pixels")
    print(f"  rms bias          = {np.sqrt(np.mean(bias_naive**2)):.6e} pixels")

    print("\nPixel-integrated fit:")
    print(f"  peak-to-peak bias = {np.ptp(bias_integrated):.6e} pixels")
    print(f"  rms bias          = {np.sqrt(np.mean(bias_integrated**2)):.6e} pixels")

    return {
        "phases": phases,
        "bias_naive": bias_naive,
        "bias_integrated": bias_integrated,
        "fitted_center_naive": fitted_center_naive,
        "fitted_center_integrated": fitted_center_integrated,
    }


if __name__ == "__main__":
    results = run_pixel_phase_fit_test(
        gamma_true=0.5,
        n_pixels=11,
        n_phase=201,
        noise_sigma=0.0,
    )