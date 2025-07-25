import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

# --- Helper Function ---
def extract_snr_and_energy(series_path: str, reference: str):
    with open(series_path, "rb") as f:
        series_list = pickle.load(f)  # list[MeasurementSeries]

    all_energies = []
    all_snrs = []

    for series in series_list:
        calibration = series.calibration_data

        def get_freq(px):
            if reference == "left":
                return calibration.left_pixel.get_freq(px)
            elif reference == "right":
                return calibration.right_pixel.get_freq(px)
            elif reference == "distance":
                return calibration.peak_distance.get_freq(px)
            else:
                raise ValueError("Invalid reference")

        px_values = []
        for mp in series.measurements:
            fs = mp.fitting_results
            if not fs.is_success:
                continue

            if reference == "left":
                px = fs.left_peak_center_px
            elif reference == "right":
                px = fs.right_peak_center_px
            else:
                px = fs.inter_peak_distance

            px_values.append(px)

        if len(px_values) < 2:
            continue

        # Compute frequency mean/std from px
        px_values = np.array(px_values)
        freqs = get_freq(px_values)
        snr = np.mean(freqs) / np.std(freqs)

        # Exposure time and energy
        t_exp = series.measurements[0].andor_exposure_settings.exposure_time_s
        energy_mJ = (power_mW * 1e-3) * t_exp * 1e3  # Convert to mJ

        all_energies.append(energy_mJ)
        all_snrs.append(snr)

    return all_energies, all_snrs

# --- Plotting ---
plt.figure(figsize=(8, 6))

for label, path in series_paths.items():
    if not Path(path).exists():
        print(f"Missing file: {path}")
        continue

    energies, snrs = extract_snr_and_energy(path, reference)
    if energies:
        plt.plot(energies, snrs, 'o-', label=label)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Illumination Energy (mJ)")
plt.ylabel(f"SNR (mean / std) – {reference} peak")
plt.title("SNR vs Illumination Energy")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.xlim(0.01, 10)     # Set x-axis from 0.01 to 10 mJ
plt.ylim(1, 1e4)       # Set y-axis from 1 to 1000 (SNR)
plt.tight_layout()



# --- Second Plot: std(frequency) only ---
plt.figure(figsize=(8, 6))

for label, path in series_paths.items():
    if not Path(path).exists():
        continue

    with open(path, "rb") as f:
        series_list = pickle.load(f)

    stds = []
    energies = []

    for series in series_list:
        calibration = series.calibration_data

        def get_freq(px):
            if reference == "left":
                return calibration.left_pixel.get_freq(px)
            elif reference == "right":
                return calibration.right_pixel.get_freq(px)
            elif reference == "distance":
                return calibration.peak_distance.get_freq(px)
            else:
                raise ValueError("Invalid reference")

        px_values = [
            mp.fitting_results.left_peak_center_px if reference == "left"
            else mp.fitting_results.right_peak_center_px if reference == "right"
            else mp.fitting_results.inter_peak_distance
            for mp in series.measurements if mp.fitting_results.is_success
        ]

        if len(px_values) < 2:
            continue

        freqs = get_freq(np.array(px_values))
        std = np.std(freqs)
        t_exp = series.measurements[0].andor_exposure_settings.exposure_time_s
        energy_mJ = (power_mW * 1e-3) * t_exp * 1e3

        energies.append(energy_mJ)
        stds.append(std * 1e3)

    if energies:
        plt.plot(energies, stds, 'o-', label=label)

plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Illumination Energy (mJ)")
plt.ylabel(f"Std. Dev. of Frequency (MHz) – {reference} peak")
plt.title("Frequency Standard Deviation vs Illumination Energy")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.xlim(0.4, 4)     # Set x-axis from 0.01 to 10 mJ
# plt.ylim(1, 50)
plt.tight_layout()



# --- Histogram: Plastic, gain 200 @ 0.2 s ---
target_file = "2025-5-26/series3.pkl"
target_exposure = 0.2  # seconds

with open(target_file, "rb") as f:
    series_list = pickle.load(f)

for series in series_list:
    t_exp = series.measurements[0].andor_exposure_settings.exposure_time_s
    if abs(t_exp - target_exposure) > 1e-3:
        continue

    calibration = series.calibration_data

    def get_freq(px):
        if reference == "left":
            return calibration.left_pixel.get_freq(px)
        elif reference == "right":
            return calibration.right_pixel.get_freq(px)
        elif reference == "distance":
            return calibration.peak_distance.get_freq(px)
        else:
            raise ValueError("Invalid reference")

    px_values = [
        mp.fitting_results.left_peak_center_px if reference == "left"
        else mp.fitting_results.right_peak_center_px if reference == "right"
        else mp.fitting_results.inter_peak_distance
        for mp in series.measurements if mp.fitting_results.is_success
    ]

    if not px_values:
        print("No successful measurements found for histogram.")
        break

    freqs = get_freq(np.array(px_values))  # Convert GHz to MHz

    # Plot histogram
    plt.figure(figsize=(7, 5))
    plt.hist(freqs, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel("Measured Frequency (GHz)")
    plt.ylabel("Count")
    plt.title(f"Histogram – Plastic, Gain 200 @ {target_exposure:.1f} s ({reference} peak)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    break  # Only plot one matching series


with open(target_file, "rb") as f:
    series_list = pickle.load(f)

# Find the series with the correct exposure time
target_series = next(
    (s for s in series_list if abs(s.measurements[0].andor_exposure_settings.exposure_time_s - target_exposure) < 1e-3),
    None
)

if target_series is None:
    print("No matching series found for target exposure time.")
else:
    calibration = target_series.calibration

    def get_freq(px):
        if reference == "left":
            return calibration.left_pixel.get_freq(px)
        elif reference == "right":
            return calibration.right_pixel.get_freq(px)
        elif reference == "distance":
            return calibration.peak_distance.get_freq(px)
        else:
            raise ValueError("Invalid reference")

    red_slines = []
    blue_slines = []
    x_pixels_ref = None

    for mp in target_series.measurements:
        fs = mp.fitting_results
        if not fs.is_success:
            continue

        if reference == "left":
            px = fs.left_peak_center_px
        elif reference == "right":
            px = fs.right_peak_center_px
        else:
            px = fs.inter_peak_distance

        freq = get_freq(px)
        if x_pixels_ref is None:
            x_pixels_ref = fs.x_pixels  # Assume all x_pixels are the same

        if freq < 5.08:
            red_slines.append(fs.sline)
        else:
            blue_slines.append(fs.sline)

    # --- Plot mean s-line values only ---
    plt.figure(figsize=(10, 5))

    if red_slines:
        red_mean = np.mean(np.vstack(red_slines), axis=0)
        plt.plot(x_pixels_ref, red_mean, label="Red (< 4.9 GHz)", color="red", linewidth=2)

    if blue_slines:
        blue_mean = np.mean(np.vstack(blue_slines), axis=0)
        plt.plot(x_pixels_ref, blue_mean, label="Blue (≥ 4.9 GHz)", color="blue", linewidth=2)

    plt.xlabel("Pixel")
    plt.ylabel("Mean S-Line Intensity (a.u.)")
    plt.title(f"Mean S-Lines by Frequency Group – {reference} peak")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

