import numpy as np
import matplotlib.pyplot as plt
from brillouinDAQ.utils.brillouin_spectrum_fitting import fitSpectrum  # <- Import your new fitting function


# --- Generate a synthetic test spectrum with 1 "merged" peak ---

def generate_single_brillouin_peak(length=200, center=100, width=5, height=1000, noise_level=20):
    x = np.arange(length)

    # Single Lorentzian peak
    lorentzian = height * (0.5 * width) ** 2 / ((x - center) ** 2 + (0.5 * width) ** 2)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, size=x.shape)

    # Add baseline
    baseline = 100  # counts

    spectrum = lorentzian + baseline + noise
    return x, spectrum


# --- Create data ---
x, sline = generate_single_brillouin_peak()

# --- Fit the spectrum ---
interpeak_distance, fitted_spectrum = fitSpectrum(sline)

# --- Plot the results ---
plt.figure(figsize=(8, 6))

plt.plot(x, sline, 'k.', label="Synthetic Data (Noisy)")
plt.plot(x, fitted_spectrum, 'r--', label="2-Lorentzian Fit")

if not np.isnan(interpeak_distance):
    plt.title(f"Fit Successful | Interpeak Distance = {interpeak_distance:.2f} px")
else:
    plt.title("Fit Failed (NaN)")

plt.xlabel("Pixel")
plt.ylabel("Intensity (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
