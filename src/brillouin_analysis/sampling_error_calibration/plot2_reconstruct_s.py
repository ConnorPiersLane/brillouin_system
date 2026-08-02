import numpy as np
import matplotlib.pyplot as plt

from helper_functions import gaussian, rect_func, sinc_cyc


def reconstruct_from_Sf(sigma=0.5, dpx=0.3, dx=0.002, xmax=6.0):
    """
    Demonstrates reconstruction:
        s(x) = inverse FT of S(f) over [-0.5, 0.5]
    """

    # -----------------------
    # grid
    # -----------------------
    x = np.arange(-xmax, xmax, dx)
    N = len(x)

    f = np.fft.fftfreq(N, d=dx)

    # -----------------------
    # original signal
    # -----------------------
    g = gaussian(x, sigma, center=dpx)
    rect = rect_func(x, width=1.0)
    h = np.convolve(g, rect, mode='same') * dx

    # -----------------------
    # sample
    # -----------------------
    n = np.arange(-5, 6)
    sample_idx = np.round((n - x[0]) / dx).astype(int)
    s = h[sample_idx]

    # -----------------------
    # build sampled impulse train
    # -----------------------
    s_train = np.zeros_like(x)
    s_train[sample_idx] = s / dx   # delta approximation

    # -----------------------
    # FFT → S(f)
    # -----------------------
    S = np.fft.fft(s_train)

    # -----------------------
    # keep only Nyquist band
    # -----------------------
    mask = np.abs(f) <= 0.5
    S_band = np.zeros_like(S, dtype=complex)
    S_band[mask] = S[mask]

    # -----------------------
    # inverse FT → reconstructed signal
    # -----------------------

    # instead of
    # h_rec = np.real(np.fft.ifft(S_band))
    # use shannon-nyquist interpolation formula
    h_rec = np.zeros_like(x, dtype=float)
    for ni, si in zip(n, s):
        h_rec += si * sinc_cyc(x - ni)

    # -----------------------
    # Plot
    # -----------------------
    plt.figure(figsize=(10, 6))

    plt.plot(x, h, label="True h(x) = g * rect", linewidth=2)
    plt.plot(x, h_rec, '--', label="Reconstructed from S(f)", linewidth=2)

    plt.stem(n, s, linefmt='C1-', markerfmt='C1o', basefmt='k-', label="samples")

    plt.title("Reconstruction from S(f) (Nyquist band only)")
    plt.xlabel("x (pixels)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.show()
reconstruct_from_Sf()