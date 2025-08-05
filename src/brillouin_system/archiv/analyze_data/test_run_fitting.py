import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from brillouin_system.my_dataclasses.measurements import MeasurementSeries
from brillouin_system.utils.brillouin_spectrum_fitting import get_fitted_spectrum_from_sline_only_lorentzian

# --- Parameters ---
reference = "distance"  # Choose from "left", "right", or "distance"
power_mW = 5.0      # Illumination power

# --- Input Files ---
series_paths = {
    "Plastic, gain 200": "2025-5-26/series1.pkl",
    "Plastic, gain 50":  "2025-5-26/series2.pkl",
    "Water, gain 200":   "2025-5-26/series3.pkl",
    "Water, gain 50":    "2025-5-26/series4.pkl"
}

# --- Histogram: Plastic, gain 200 @ 0.2 s ---
target_file = "2025-5-26/series3.pkl"
target_exposure = 0.6  # seconds

with open(target_file, "rb") as f:
    series_list = pickle.load(f)

for series in series_list:
    t_exp = series.measurements[0].andor_camera_info.exposure_time_s
    if abs(t_exp - target_exposure) > 1e-3:
        continue
    else:
        s: MeasurementSeries = series
        break

slines = [mp.fitting_results.sline for mp in s.measurements]

cali = s.calibration_data.peak_distance

fitted_spectras = [get_fitted_spectrum_from_sline_only_lorentzian(sline, None, False) for sline in slines]
freqs = [cali.get_freq(fs.inter_peak_distance) for fs in fitted_spectras]

[print(fs.inter_peak_distance) for fs in fitted_spectras]
distances=[fs.inter_peak_distance for fs in fitted_spectras]
print(max(distances))
print(min(distances))



# Plot histogram
plt.figure(figsize=(7, 5))
plt.hist(freqs, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
plt.xlabel("Measured Frequency (GHz)")
plt.ylabel("Count")
plt.title(f"Histogram – Plastic, Gain 200 @ {target_exposure:.1f} s ({reference} peak)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()



# Find indices of min and max distances
min_idx = np.argmin(distances)
max_idx = np.argmax(distances)

# Extract corresponding slines
min_sline = slines[min_idx]
max_sline = slines[max_idx]

# Plot both slines on the same axes
plt.figure(figsize=(8, 5))
plt.plot(min_sline, label=f"Min Distance ({distances[min_idx]:.2f} px)")
plt.plot(max_sline, label=f"Max Distance ({distances[max_idx]:.2f} px)")
plt.xlabel("Pixel")
plt.ylabel("Intensity")
plt.title("Slines – Min vs Max Inter-Peak Distance")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
