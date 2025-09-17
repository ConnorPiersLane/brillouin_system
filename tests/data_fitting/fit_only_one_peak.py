import numpy as np
import matplotlib.pyplot as plt

from brillouin_system.archiv.fitting import get_fitted_spectrum_lorentzian


# <- Import your new fitting function


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
fitted_spectrum = get_fitted_spectrum_lorentzian(sline, is_reference_mode=False)

# --- Plot the results ---
plt.figure(figsize=(8, 6))

plt.plot(x, sline, 'k.', label="Synthetic Data (Noisy)")
plt.plot(fitted_spectrum.x_fit_refined, fitted_spectrum.y_fit_refined, 'r--', label="2-Lorentzian Fit")

if not np.isnan(fitted_spectrum.inter_peak_distance):
    plt.title(f"Fit Successful | Interpeak Distance = {fitted_spectrum.inter_peak_distance:.2f} px")
else:
    plt.title("Fit Failed (NaN)")

plt.xlabel("Pixel")
plt.ylabel("Intensity (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
