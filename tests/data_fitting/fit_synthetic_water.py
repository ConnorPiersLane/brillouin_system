import numpy as np
import matplotlib.pyplot as plt
from brillouinDAQ.devices.acquisition.brillouin_spectrum_fitting import fitSpectrum


# --- Step 1: Generate synthetic data ---
def generate_water_signal(x, peaks, width=3, noise_level=200, baseline=1000):
    y = np.full_like(x, baseline, dtype=float)
    for pos, height in peaks:
        y += height * np.exp(-0.5 * ((x - pos) / width) ** 2)
    noise = np.random.normal(0, noise_level, size=x.shape)
    return y + noise


x = np.arange(0, 160)
test_data_sets = []

for i in range(5):
    np.random.seed(i)
    peaks = [(np.random.randint(40, 60), np.random.randint(2000, 5000)),
             (np.random.randint(100, 120), np.random.randint(2000, 5000))]
    y = generate_water_signal(x, peaks)
    test_data_sets.append(y)

# --- Step 2: Fit using your function and plot ---
fig, axs = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
for i, sline in enumerate(test_data_sets):
    inter_peak_dist, fitted = fitSpectrum(sline)

    axs[i].plot(x, sline, 'o', markersize=4, label="Synthetic Data (Points)")
    axs[i].plot(x, fitted, '-', linewidth=2, label="2-Lorentzian Fit")
    axs[i].set_title(f"Dataset {i+1} | Interpeak Distance: {inter_peak_dist:.2f}")
    axs[i].set_ylabel("Counts")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Pixel Number")
plt.tight_layout()
plt.show()

