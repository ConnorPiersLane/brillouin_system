import numpy as np
import matplotlib.pyplot as plt
from brillouinDAQ.devices.acquisition.brillouin_spectrum_fitting import fitSpectrum

# Step 1: Generator with sloped background
def generate_plastic_with_background(x, peaks, width=2.5, noise_level=500, baseline=1200):
    """Synthetic plastic-like Brillouin signal with sloped baseline."""
    background = baseline + 0.015 * (x - np.mean(x))**2  # quadratic baseline
    signal = np.copy(background)
    for pos, height in peaks:
        signal += height * np.exp(-0.5 * ((x - pos) / width)**2)
    noise = np.random.normal(0, noise_level, size=x.shape)
    return signal + noise

# Step 2: Generate 5 synthetic datasets
x = np.arange(0, 160)
plastic_data_sets = []

for i in range(5):
    np.random.seed(100 + i)
    peaks = [(np.random.randint(70, 80), np.random.randint(2500, 4000)),
             (np.random.randint(85, 95), np.random.randint(2500, 4000))]
    y = generate_plastic_with_background(x, peaks)
    plastic_data_sets.append(y)

# Step 3: Fit and plot
fig, axs = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
for i, sline in enumerate(plastic_data_sets):
    inter_peak_dist, fitted = fitSpectrum(sline)

    axs[i].scatter(x, sline, s=10, color='black', label="Synthetic Plastic Signal")
    axs[i].plot(x, fitted, '--r', label="2-Lorentzian Fit")
    axs[i].set_title(f"Plastic Dataset {i+1} | Interpeak Distance: {inter_peak_dist:.2f}")
    axs[i].set_ylabel("Counts")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Pixel Number")
plt.tight_layout()
plt.show()
