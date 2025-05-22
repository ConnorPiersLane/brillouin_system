import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from brillouinDAQ.my_dataclasses.measurement_data import MeasurementData
from brillouinDAQ.utils.brillouin_spectrum_fitting import find_brillouin_peak_locations, select_top_two_peaks

# === User Input ===
# Path to the pickle file containing the list of series (list[list[MeasurementData]])
# Path to current script
current_file = Path(__file__).resolve()

# Go one level up and into the 'apps' folder
data_folder = r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2025-5-19"




# Index of the series you want to analyze (0-based)
series_index = 0  # ← change this

# Example: target file inside the apps folder
pkl_path = data_folder + "\gain_200_5_pixel_rows.pkl"
# === Load Data ===
with open(pkl_path, "rb") as f:
    all_series_200: list[list[MeasurementData]] = pickle.load(f)

# Example: target file inside the apps folder
pkl_path = data_folder + "\gain_50_5_pixel_rows.pkl"
with open(pkl_path, "rb") as f:
    all_series_50: list[list[MeasurementData]] = pickle.load(f)

print(f"[✓] Loaded {len(all_series_200)} measurement series.")
selected_series: list[MeasurementData] = all_series_200[series_index]
print(f"[→] Analyzing series #{series_index} with {len(selected_series)} measurements.")


# === Constants ===
laser_power_mW = 20  # Your input power in mW

# === Convert exposure time to optical energy in mJ ===
def compute_energy_mJ(expo_times: list[float], power_mW: float) -> list[float]:
    return [t * power_mW for t in expo_times]


def extract_peak_max_intensities(selected_series: list[MeasurementData], side: str) -> list[float]:
    """
    For each MeasurementData in the series, extract the max pixel value from the left Brillouin peak.

    Parameters:
        selected_series (List[MeasurementData]): A series of measurements.
        side: 'left' or 'right'

    Returns:
        List[float]: Max pixel values of the left peak from each spectrum.
    """
    peak_maxima = []

    for measurement in selected_series:
        sline = measurement.fitting_results.sline  # 1D spectrum
        pk_ind, pk_info = find_brillouin_peak_locations(sline)
        pk_ind, _ = select_top_two_peaks(pk_ind, pk_info)

        if len(pk_ind) >= 1:
            if side == 'left':
                left_idx = int(np.min(pk_ind))  # take the leftmost peak index
                peak_max = sline[left_idx]
            elif side == 'right':
                r_idx = int(np.max(pk_ind))  # take the leftmost peak index
                peak_max = sline[r_idx]
            else:
                raise ValueError(f"side={side} incorrect value")
        else:
            peak_max = 0  # no peak found

        peak_maxima.append(peak_max)

    return peak_maxima




def analyze_snr(int_function: Callable, selected_series: list[MeasurementData], side: str) -> tuple[float, float, float]:
    intensities = int_function(selected_series, side)
    mean, snr = np.mean(intensities), np.std(intensities)
    return mean/snr, mean, snr


# peaks = extract_peak_max_intensities(selected_series=selected_series, side='left')
# plot_peak_maxima_histogram(peaks, 'left')

snrs_left = []
expo_times_left = []
for series in all_series_200:
    snr, _, _ = analyze_snr(extract_peak_max_intensities, series, side='left')
    snrs_left.append(snr)
    expo_times_left.append(series[0].camera_settings.exposure_time_s)

snrs_r = []
expo_times_r = []
for series in all_series_200:
    snr, _, _ = analyze_snr(extract_peak_max_intensities, series, side='right')
    snrs_r.append(snr)
    expo_times_r.append(series[0].camera_settings.exposure_time_s)

snrs_left_50 = []
expo_times_left_50 = []
for series in all_series_50:
    snr, _, _ = analyze_snr(extract_peak_max_intensities, series, side='left')
    snrs_left_50.append(snr)
    expo_times_left_50.append(series[0].camera_settings.exposure_time_s)

snrs_r_50 = []
expo_times_r_50 = []
for series in all_series_50:
    snr, _, _ = analyze_snr(extract_peak_max_intensities, series, side='right')
    snrs_r_50.append(snr)
    expo_times_r_50.append(series[0].camera_settings.exposure_time_s)






# === Plot log-log SNR vs Energy ===
# === Compute energies ===
energy_200_left = compute_energy_mJ(expo_times_left, laser_power_mW)
energy_200_right = compute_energy_mJ(expo_times_r, laser_power_mW)
energy_50_left = compute_energy_mJ(expo_times_left_50, laser_power_mW)
energy_50_right = compute_energy_mJ(expo_times_r_50, laser_power_mW)
plt.figure(figsize=(8, 6))

plt.loglog(energy_200_left, snrs_left, marker='o', linestyle='-', label='Gain=200 Left Pixel')
plt.loglog(energy_200_right, snrs_r, marker='s', linestyle='-', label='Gain=200 Right Pixel')
plt.loglog(energy_50_left, snrs_left_50, marker='^', linestyle='--', label='Gain=50 Left Pixel')
plt.loglog(energy_50_right, snrs_r_50, marker='v', linestyle='--', label='Gain=50 Right Pixel')

# Set axis scales and limits to match Fig. S4(a)
plt.xlim(1e-1, 0.2e2)
plt.ylim(1e0, 1e2)

# Labels, grid, legend
plt.xlabel("Optical Energy (mJ)")
plt.ylabel("SNR")
plt.title("SNR vs Optical Energy (20 mW laser)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
# plt.show()

side = 'left'
peaks = extract_peak_max_intensities(selected_series=all_series_50[3], side=side)
plt.figure(figsize=(6, 4))
plt.hist(peaks, bins=30, color='skyblue', edgecolor='black')
plt.title(f"{side.capitalize()} Peak Max Intensities")
plt.xlabel("Max Pixel Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.xlim(0, 4000)  # Set x-axis limits
plt.ylim(0, 15)
plt.tight_layout()



def get_brillouin_values(series: list[MeasurementData]) -> list[float]:
    return [s.fitting_results.freq_shift_ghz for s in series]

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# Extract Brillouin frequency shifts from selected series
brillouin_shifts = get_brillouin_values(selected_series)

# Compute mean and standard deviation
mean_shift = np.mean(brillouin_shifts)
std_shift = np.std(brillouin_shifts)

# Plot histogram
plt.figure(figsize=(6, 4))
counts, bins, _ = plt.hist(brillouin_shifts, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.6)

# Plot Gaussian fit
x = np.linspace(min(bins), max(bins), 1000)
pdf = norm.pdf(x, mean_shift, std_shift)
plt.plot(x, pdf, 'r--', linewidth=2, label=f'Gaussian Fit\nμ = {mean_shift:.4f} GHz\nσ = {std_shift:.4f} GHz')

# Add labels, legend, and formatting
plt.title("Histogram of Brillouin Frequency Shifts with Gaussian Fit")
plt.xlabel("Brillouin Frequency Shift (GHz)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


