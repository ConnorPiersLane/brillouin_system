import numpy as np
import matplotlib.pyplot as plt
from brillouinDAQ.devices.acquisition.brillouin_spectrum_fitting import fitSpectrum


def generate_plastic_signal(x, peaks, width=2, noise_level=150, baseline=1200):
    """Synthetic Brillouin spectrum for plastic-like material."""
    y = np.full_like(x, baseline, dtype=float)
    for pos, height in peaks:
        y += height * np.exp(-0.5 * ((x - pos) / width) ** 2)
    noise = np.random.normal(0, noise_level, size=x.shape)
    return y + noise


# Setup for synthetic plastic spectra
x = np.arange(0, 160)
plastic_data_sets = []

for i in range(5):
    np.random.seed(10 + i)
    # Closer peaks for plastic
    peaks = [(np.random.randint(70, 80), np.random.randint(2000, 4500)),
             (np.random.randint(85, 95), np.random.randint(2000, 4500))]
    y = generate_plastic_signal(x, peaks)
    plastic_data_sets.append(y)

# Plot with fit
fig, axs = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
for i, sline in enumerate(plastic_data_sets):
    inter_peak_dist, fitted = fitSpectrum(sline)

    axs[i].scatter(x, sline, s=10, color='black', label="Synthetic Plastic Signal")
    axs[i].plot(x, fitted, '--r', label="2-Lorentzian Fit")
    axs[i].set_title(f"Plastic Dataset {i + 1} | Interpeak Distance: {inter_peak_dist:.2f}")
    axs[i].set_ylabel("Counts")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Pixel Number")
plt.tight_layout()
plt.show()
