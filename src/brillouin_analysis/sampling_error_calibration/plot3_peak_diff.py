import numpy as np
import matplotlib.pyplot as plt

from src.helper_functions import estimate_center_from_peak, estimate_center_from_centroid, pixel_samples_lorentzian, \
    sinc_reconstruct, lorentzian, pixel_integrated_lorentzian_continuous


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


# ------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------

def center_bias_walkthrough(
    sigma=1.2,
    n_pixels=11,
    d_grid=np.linspace(0.0, 1.0, 201, endpoint=False),
    d_example=0.60,
    x_halfwidth=6.0,
    dx=0.001,
    estimator="peak",
):
    """
    Study center bias over one pixel walk.

    Parameters
    ----------
    sigma : float
        lorentzianian std in pixels.
    n_pixels : int
        Number of integer pixel samples used, centered around 0.
    d_grid : array
        True centers to test over one pixel, usually [0,1).
    d_example : float
        One example center to plot in detail.
    x_halfwidth : float
        Plot / reconstruction domain is [-x_halfwidth, x_halfwidth].
    dx : float
        Fine x-grid spacing.
    estimator : str
        'peak' or 'centroid'

    Returns
    -------
    results : dict
        Dictionary containing d_grid, x_rec, bias, and example curves.
    """
    if n_pixels % 2 == 0:
        raise ValueError("n_pixels should be odd so the sample window is symmetric.")

    # Integer sample positions
    half = n_pixels // 2
    n = np.arange(-half, half + 1, dtype=float)

    # Fine continuous grid
    x = np.arange(-x_halfwidth, x_halfwidth + dx, dx)

    # Center estimator
    if estimator == "peak":
        center_estimator = estimate_center_from_peak
    elif estimator == "centroid":
        center_estimator = estimate_center_from_centroid
    else:
        raise ValueError("estimator must be 'peak' or 'centroid'.")

    # Sweep one pixel
    x_rec_list = []
    bias_list = []

    for d in d_grid:
        s = pixel_samples_lorentzian(n, sigma, d)
        h_rec = sinc_reconstruct(x, n, s)
        x_rec = center_estimator(x, h_rec)
        x_rec_list.append(x_rec)
        bias_list.append(x_rec - d)

    x_rec_arr = np.array(x_rec_list)
    bias_arr = np.array(bias_list)

    # Example curves
    g_true_ex = lorentzian(x, sigma, d_example)
    h_true_ex = pixel_integrated_lorentzian_continuous(x, sigma, d_example)
    s_ex = pixel_samples_lorentzian(n, sigma, d_example)
    h_rec_ex = sinc_reconstruct(x, n, s_ex)
    x_rec_ex = center_estimator(x, h_rec_ex)
    bias_ex = x_rec_ex - d_example

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Center bias from sinc reconstruction\n"
        f"sigma={sigma:.3f} px, n_pixels={n_pixels}, estimator={estimator}",
        fontsize=14
    )

    # Example: original lorentzianian and reconstructed signal
    ax = axes[0, 0]
    ax.plot(x, g_true_ex, label="Original lorentzianian g(x)")
    ax.plot(x, h_true_ex, label="True pixel-integrated h(x)=g*rect")
    ax.plot(x, h_rec_ex, "--", label="Reconstructed h_rec(x)")
    ax.axvline(d_example, color="k", linestyle=":", label=f"true center = {d_example:.3f}")
    ax.axvline(x_rec_ex, color="r", linestyle="--", label=f"reconstructed center = {x_rec_ex:.3f}")
    ax.set_title("One example walkthrough")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()

    # Example: samples
    ax = axes[0, 1]
    ax.plot(x, h_true_ex, alpha=0.7, label="True h(x)")
    markerline, stemlines, baseline = ax.stem(n, s_ex, linefmt='C1-', markerfmt='C1o', basefmt='k-')
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(markerline, markersize=6)
    ax.set_title(f"Discrete pixel samples for d={d_example:.3f}")
    ax.set_xlabel("Pixel center")
    ax.set_ylabel("Sample value")
    ax.grid(True)
    ax.legend()

    # Center estimate over one pixel
    ax = axes[1, 0]
    ax.plot(d_grid, x_rec_arr, label=r"$x_{\rm rec}(d)$")
    ax.plot(d_grid, d_grid, "k--", label=r"$x_{\rm true}=d$")
    ax.set_title("Recovered center over one-pixel walk")
    ax.set_xlabel("True center d (pixels)")
    ax.set_ylabel("Recovered center")
    ax.grid(True)
    ax.legend()

    # Bias
    ax = axes[1, 1]
    ax.plot(d_grid, bias_arr, label=r"$\Delta x(d)=x_{\rm rec}(d)-d$")
    ax.axhline(0.0, color="k", linestyle="--")
    ax.set_title("Center bias over one pixel")
    ax.set_xlabel("True center d (pixels)")
    ax.set_ylabel("Bias (pixels)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

    print(f"Example d = {d_example:.6f}")
    print(f"Recovered center = {x_rec_ex:.6f}")
    print(f"Bias = {bias_ex:.6e} pixels")

    return {
        "d_grid": d_grid,
        "x_rec": x_rec_arr,
        "bias": bias_arr,
        "x": x,
        "n": n,
        "g_true_example": g_true_ex,
        "h_true_example": h_true_ex,
        "s_example": s_ex,
        "h_rec_example": h_rec_ex,
        "x_rec_example": x_rec_ex,
        "bias_example": bias_ex,
    }

# ------------------------------------------------------------
# Example use
# ------------------------------------------------------------

results = center_bias_walkthrough(
    sigma=0.7,
    n_pixels=11,
    d_grid=np.linspace(0.0, 1.0, 201, endpoint=False),
    d_example=0.25,
    dx=0.001,
    estimator="peak",   # try also "centroid"
)