import numpy as np
import matplotlib.pyplot as plt

from helper_functions import gaussian, rect_func, sinc_cyc


def plot_gaussian_pipeline_with_phase(
    sigma=0.5, dpx=0.0, pixel_width=1.0, dx=0.002, xmax=6.0, fmax=2.5
):
    """
    Plot the full pipeline:
      1) Gaussian g(x)
      2) rect(x)
      3) h(x) = g * rect
      4) sampled h(x) at integer pixels
      5) magnitude responses G(f), sinc(f), H(f)=G(f)*sinc(f)
      5.2) phase responses of G(f), sinc(f), H(f)
      6) replicated spectrum magnitude after sampling
      6.2) replicated spectrum phase after sampling
    """

    # -----------------------
    # Continuous spatial grid
    # -----------------------
    x = np.arange(-xmax, xmax + dx, dx)

    # -----------------------
    # Step 1: Gaussian
    # -----------------------
    g = gaussian(x, sigma, center=dpx)

    # -----------------------
    # Step 2: Pixel aperture
    # -----------------------
    rect = rect_func(x, width=pixel_width)

    # -----------------------
    # Step 3: Pixel integration
    # h(x) = g * rect
    # -----------------------
    h = np.convolve(g, rect, mode='same') * dx

    # -----------------------
    # Step 4: Sample at integer pixel centers
    # -----------------------
    n = np.arange(-5, 6, 1)
    sample_idx = np.round((n - x[0]) / dx).astype(int)
    s = h[sample_idx]

    # -----------------------
    # Step 5: Frequency-domain expressions
    # Use complex G(f) so the shift dpx appears in phase
    # -----------------------
    f = np.linspace(-fmax, fmax, 4000)

    G_mag = np.exp(-2 * np.pi**2 * sigma**2 * f**2)
    G = G_mag * np.exp(-1j * 2 * np.pi * f * dpx)

    R = sinc_cyc(f).astype(complex)
    H = G * R

    # For cleaner phase plots, hide phase where magnitude is tiny
    phase_thresh = 1e-6
    G_phase = np.angle(G)
    R_phase = np.angle(R)
    H_phase = np.angle(H)

    G_phase[np.abs(G) < phase_thresh] = np.nan
    R_phase[np.abs(R) < phase_thresh] = np.nan
    H_phase[np.abs(H) < phase_thresh] = np.nan

    # -----------------------
    # Step 6: Spectrum after sampling
    # S(f) = sum_k H(f-k)
    # -----------------------
    S = np.zeros_like(f, dtype=complex)
    kmin, kmax = -4, 4
    H_replicas = []

    for k in range(kmin, kmax + 1):
        fk = f - k
        Gk_mag = np.exp(-2 * np.pi**2 * sigma**2 * fk**2)
        Gk = Gk_mag * np.exp(-1j * 2 * np.pi * fk * dpx)
        Rk = sinc_cyc(fk).astype(complex)
        Hk = Gk * Rk
        H_replicas.append(Hk)
        S += Hk

    S_phase = np.angle(S)
    S_phase[np.abs(S) < phase_thresh] = np.nan

    # -----------------------
    # Plotting
    # -----------------------
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    fig.suptitle("Gaussian + Pixel Integration + Sampling + Phase", fontsize=16)

    # 1) Gaussian
    ax = axes[0, 0]
    ax.plot(x, g, label=r"$g(x)$ Gaussian")
    ax.axvline(dpx, color='k', linestyle='--', alpha=0.6, label=f"center = {dpx:.2f}")
    ax.set_title("Step 1: Continuous Gaussian signal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()

    # 2) Rect aperture
    ax = axes[0, 1]
    ax.plot(x, rect, label=r"$\mathrm{rect}(x)$ pixel aperture")
    ax.set_title("Step 2: Pixel aperture (rect)")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True)
    ax.legend()

    # 3) Convolution result
    ax = axes[0, 2]
    ax.plot(x, g, '--', alpha=0.7, label=r"$g(x)$")
    ax.plot(x, h, label=r"$h(x)=g * \mathrm{rect}$")
    ax.set_title("Step 3: Pixel-integrated continuous signal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()

    # 4) Sampled signal
    ax = axes[1, 0]
    ax.plot(x, h, alpha=0.7, label=r"$h(x)$ continuous integrated signal")
    markerline, stemlines, baseline = ax.stem(
        n, s, linefmt='C1-', markerfmt='C1o', basefmt='k-'
    )
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(markerline, markersize=6)
    ax.set_title("Step 4: Sampled at pixel centers")
    ax.set_xlabel("Pixel index / position")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()

    # 5) Presampled frequency response magnitude
    ax = axes[1, 1]
    ax.plot(f, np.abs(G), label=r"$|G(f)|$")
    ax.plot(f, np.abs(R), label=r"$|\mathrm{sinc}(f)|$")
    ax.plot(f, np.abs(H), linewidth=2, label=r"$|H(f)|=|G(f)\,\mathrm{sinc}(f)|$")
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.6, label="Nyquist")
    ax.axvline(-0.5, color='k', linestyle='--', alpha=0.6)
    ax.set_title("Step 5: Presampled frequency magnitude")
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.25, 1.05)
    ax.grid(True)
    ax.legend()

    # 6) Spectrum replicas after sampling magnitude
    ax = axes[1, 2]
    ax.plot(f, np.abs(H), '--', alpha=0.7, label="Presampled |H(f)|")
    for Hk in H_replicas:
        ax.plot(f, np.abs(Hk), alpha=0.35)
    ax.plot(f, np.abs(S), linewidth=2, label=r"$|S(f)|=|\sum_k H(f-k)|$")
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.6, label="Nyquist")
    ax.axvline(-0.5, color='k', linestyle='--', alpha=0.6)
    ax.set_title("Step 6: Replicated spectrum magnitude")
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True)
    ax.legend()

    # 5.2) Presampled frequency phase
    ax = axes[2, 1]
    ax.plot(f, G_phase, label=r"$\angle G(f)$")
    ax.plot(f, R_phase, label=r"$\angle \mathrm{sinc}(f)$")
    ax.plot(f, H_phase, linewidth=2, label=r"$\angle H(f)$")
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.6, label="Nyquist")
    ax.axvline(-0.5, color='k', linestyle='--', alpha=0.6)
    ax.set_title("Step 5.2: Presampled frequency phase")
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("Phase (rad)")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    ax.grid(True)
    ax.legend()

    # 6.2) Replicated spectrum phase
    ax = axes[2, 2]
    for Hk in H_replicas:
        phase_k = np.angle(Hk)
        phase_k[np.abs(Hk) < phase_thresh] = np.nan
        ax.plot(f, phase_k, alpha=0.25)
    ax.plot(f, S_phase, linewidth=2, label=r"$\angle S(f)$")
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.6, label="Nyquist")
    ax.axvline(-0.5, color='k', linestyle='--', alpha=0.6)
    ax.set_title("Step 6.2: Replicated spectrum phase")
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("Phase (rad)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    ax.grid(True)
    ax.legend()

    # bottom-left: explanation panel
    ax = axes[2, 0]
    ax.axis("off")
    ax.text(
        0.0, 0.95,
        "Phase notes:\n\n"
        "• A shift by dpx gives a phase ramp:\n"
        "    G(f) -> G(f) exp(-i 2π f dpx)\n\n"
        "• |G(f)| does not change with center,\n"
        "  but phase does.\n\n"
        "• Center information lives in phase.\n\n"
        "• sinc(f) is real, so its phase is mostly 0 or π\n"
        "  depending on sign.\n\n"
        "• The sampled replicated spectrum S(f)\n"
        "  also carries this phase structure.",
        va="top",
        fontsize=11
    )

    plt.tight_layout()
    plt.show()

plot_gaussian_pipeline_with_phase(dpx=0)