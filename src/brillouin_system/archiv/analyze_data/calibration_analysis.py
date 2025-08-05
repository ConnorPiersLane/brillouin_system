import pickle
import matplotlib.pyplot as plt
import numpy as np

from brillouin_system.archiv.analyze_data.sorter import sort_fitted_spectrum_peaks
from brillouin_system.calibration.calibration import calibrate, get_calibration_fig, CalibrationResults

# --- Helper Function ---
def plot_frequency_errors(calibration_results: CalibrationResults, reference: str):
    assert reference in ["left", "right", "distance"], "Invalid reference type"

    # Choose calibration and pixel extractor
    if reference == "left":
        extract_px = lambda fs: fs.left_peak_center_px
        calibration = calibration_results.left_pixel
        y_label = "Left Peak Frequency Error (GHz)"
    elif reference == "right":
        extract_px = lambda fs: fs.right_peak_center_px
        calibration = calibration_results.right_pixel
        y_label = "Right Peak Frequency Error (GHz)"
    else:
        extract_px = lambda fs: fs.inter_peak_distance
        calibration = calibration_results.peak_distance
        y_label = "Inter-Peak Distance Frequency Error (GHz)"

    freqs = []
    freq_errors_std = []
    freq_errors_delta = []

    for freq, spectra_list in zip(calibration_results.data.freqs, calibration_results.data.cali_meas_points):
        pixels = np.array([extract_px(fs) for fs in spectra_list if fs.is_success])
        if len(pixels) == 0:
            continue

        px_mean = np.mean(pixels)
        px_std = np.std(pixels)

        dfdx = 2 * calibration.a * px_mean + calibration.b
        freq_std = abs(dfdx * px_std)

        freq_fit = calibration.get_freq(px_mean)
        freq_delta = abs(freq - freq_fit)

        freqs.append(freq)
        freq_errors_std.append(freq_std)
        freq_errors_delta.append(freq_delta)

    fig, ax = plt.subplots()
    ax.plot(freqs, [f*1e3 for f in freq_errors_std], 'o-', label="df/dpx(px_mean) x px_std")
    ax.plot(freqs, [f*1e3 for f in freq_errors_delta], 's--', label="|f_0 - f_fit|")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Frequency Error (MHz)")
    ax.set_title(f"Calibration Frequency Error Analysis ({reference})")
    ax.set_ylim(0, 35)
    ax.grid(True)
    ax.legend()




def plot_average_peak_widths_in_frequency(calibration_results: CalibrationResults):
    freqs = []
    left_means = []
    left_stds = []
    right_means = []
    right_stds = []

    for freq, spectra_list in zip(calibration_results.data.freqs, calibration_results.data.cali_meas_points):
        left_widths = []
        right_widths = []

        for fs in spectra_list:
            if not fs.is_success:
                continue

            # Derivative of calibration curve at the peak position
            df_dx_left = 2 * calibration_results.left_pixel.a * fs.left_peak_center_px + calibration_results.left_pixel.b
            df_dx_right = 2 * calibration_results.right_pixel.a * fs.right_peak_center_px + calibration_results.right_pixel.b

            # Convert pixel HWHM to GHz
            left_widths.append(abs(df_dx_left * fs.left_peak_width_px))
            right_widths.append(abs(df_dx_right * fs.right_peak_width_px))

        if left_widths and right_widths:
            freqs.append(freq)
            left_widths = np.array(left_widths)
            right_widths = np.array(right_widths)

            left_means.append(left_widths.mean())
            left_stds.append(left_widths.std())

            right_means.append(right_widths.mean())
            right_stds.append(right_widths.std())

    # Convert GHz → MHz
    left_means = [w * 1e3 for w in left_means]
    left_stds = [s * 1e3 for s in left_stds]
    right_means = [w * 1e3 for w in right_means]
    right_stds = [s * 1e3 for s in right_stds]

    # Plot
    fig, ax = plt.subplots()
    ax.errorbar(freqs, left_means, yerr=left_stds, fmt='g-', capsize=4, label="Left Peak Width (MHz)")
    ax.errorbar(freqs, right_means, yerr=right_stds, fmt='r--', capsize=4, label="Right Peak Width (MHz)")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("HWHM (MHz)")
    ax.set_title("Peak Widths in Frequency Domain (HWHM ± Std)")
    ax.grid(True)
    ax.legend()


def plot_sensitivity_comparison(calibration_results: CalibrationResults, reference: str = "left"):
    assert reference in ["left", "right"], "Reference must be 'left' or 'right'"

    # Select calibration and parameter accessors
    if reference == "left":
        calibration = calibration_results.left_pixel
        get_x0 = lambda fs: fs.left_peak_center_px
        get_A = lambda fs: fs.left_peak_amplitude
        get_gamma = lambda fs: fs.left_peak_width_px
        title = "Left Peak Sensitivity Comparison"
    else:
        calibration = calibration_results.right_pixel
        get_x0 = lambda fs: fs.right_peak_center_px
        get_A = lambda fs: fs.right_peak_amplitude
        get_gamma = lambda fs: fs.right_peak_width_px
        title = "Right Peak Sensitivity Comparison"

    a, b = calibration.a, calibration.b

    freqs = []
    sens_pi = []
    sens_deriv = []

    for freq, spectra_list in zip(calibration_results.data.freqs, calibration_results.data.cali_meas_points):
        widths_ghz = []
        Ns = []
        pixels = []

        for fs in spectra_list:
            if not fs.is_success:
                continue

            x0 = get_x0(fs)
            A = get_A(fs)
            gamma_px = get_gamma(fs)

            if None in [x0, A, gamma_px]:
                continue

            # Convert HWHM to frequency space
            df_dx = 2 * a * x0 + b
            width_ghz = abs(df_dx * gamma_px)

            # Photon count estimation
            N = np.pi * A * gamma_px
            if N <= 0:
                continue

            delta_f_pi = width_ghz / np.sqrt(N)

            widths_ghz.append(width_ghz)
            Ns.append(N)
            pixels.append(x0)

        if not pixels:
            continue

        # Average per frequency point
        freqs.append(freq)
        sens_pi.append(np.mean([w / np.sqrt(n) for w, n in zip(widths_ghz, Ns)]))

        # Propagated pixel std
        px_mean = np.mean(pixels)
        px_std = np.std(pixels)
        df_dx_mean = 2 * a * px_mean + b
        delta_f_deriv = abs(df_dx_mean * px_std)
        sens_deriv.append(delta_f_deriv)

def plot_distance_fit_and_error_with_sensitivity(calibration_results):
    # Extract frequency and inter-peak distances
    freqs = np.array(calibration_results.data.freqs)
    distances = []
    pixel_stds = []

    for spectra_list in calibration_results.data.cali_meas_points:
        px_values = [fs.inter_peak_distance for fs in spectra_list if fs.is_fitting_available]
        if px_values:
            distances.append(np.mean(px_values))
            pixel_stds.append(np.std(px_values))
        else:
            distances.append(np.nan)
            pixel_stds.append(np.nan)

    distances = np.array(distances)
    pixel_stds = np.array(pixel_stds)
    valid = ~np.isnan(distances)
    freqs = freqs[valid]
    distances = distances[valid]
    pixel_stds = pixel_stds[valid]

    # Linear fit: f = a * d + b
    a, b = np.polyfit(distances, freqs, 1)
    dist_fit = np.linspace(distances.min(), distances.max(), 200)
    freq_fit = a * dist_fit + b

    # --- Plot 1: Linear fit ---
    fig1, ax1 = plt.subplots()
    ax1.plot(distances, freqs, 'o', label='Measured Data')
    ax1.plot(dist_fit, freq_fit, '-', color='black', label='Linear Fit')
    ax1.set_xlabel("Inter-Peak Distance (px)")
    ax1.set_ylabel("Frequency (GHz)")
    ax1.set_title("Calibration: Frequency vs Inter-Peak Distance")
    ax1.grid(True)
    ax1.legend()

    # --- Plot 2: Frequency error and pixel std sensitivity ---
    freq_pred = a * distances + b
    error_mhz = np.abs(freqs - freq_pred) * 1e3  # in MHz
    sensitivity_mhz = np.abs(a) * pixel_stds * 1e3  # df/dx * std (MHz)

    fig2, ax2 = plt.subplots()
    ax2.plot(freqs, sensitivity_mhz, 'o-', label='df/dx(px_mean) × σ_px')
    ax2.plot(freqs, error_mhz, 's--', color='orange', label='|f₀ - f_fit|')
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Frequency Error / Sensitivity (MHz)")
    ax2.set_title("Calibration Frequency Error Analysis (distance)")
    ax2.set_ylim(0, 35)
    ax2.grid(True)
    ax2.legend()

def plot_slines_at_freqs(calibration_results, freqs_to_plot=(4.0, 8.0)):
    freqs = np.array(calibration_results.data.freqs)
    fig, ax = plt.subplots()

    for target_freq in freqs_to_plot:
        # Find closest match
        idx = np.argmin(np.abs(freqs - target_freq))
        spectra_list = calibration_results.data.cali_meas_points[idx]
        fs = next((s for s in spectra_list if s.is_fitting_available), None)

        if fs is None:
            print(f"No successful spectrum found near {target_freq} GHz.")
            continue

        ax.plot(fs.x_pixels, fs.sline,'.--', label=f"Sline at {freqs[idx]:.2f} GHz")

    ax.set_xlabel("Pixel")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("Sline Comparison at 4 GHz and 8 GHz")
    ax.grid(True)
    ax.legend()


# --- Load and Process Calibration Data ---
file_path = "2025-5-26/calibration3.pkl"
with open(file_path, "rb") as f:
    calibration_results: CalibrationResults = pickle.load(f)

data = calibration_results.data
data.cali_meas_points = [
    [sort_fitted_spectrum_peaks(fs) for fs in fs_list]
    for fs_list in data.cali_meas_points
]

calibration_results = calibrate(data)

# --- Calibration Visualization ---
reference_type = "distance"  # Choose 'left', 'right', or 'distance'
fig = get_calibration_fig(calibration_results, reference=reference_type)


# --- Frequency Error Analysis ---
plot_frequency_errors(calibration_results, reference=reference_type)
plot_average_peak_widths_in_frequency(calibration_results)
plot_sensitivity_comparison(calibration_results, reference='left')
plot_sensitivity_comparison(calibration_results, reference='right')
plot_distance_fit_and_error_with_sensitivity(calibration_results)
plot_slines_at_freqs(calibration_results, freqs_to_plot=(4.0,8.0))





