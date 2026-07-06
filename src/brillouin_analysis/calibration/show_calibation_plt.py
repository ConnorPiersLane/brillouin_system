import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog

from brillouin_system.calibration.calibration import (
    get_calibration_calculator_from_data,
    CalibrationData,
)
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    load_dict_from_hdf5,
    dict_to_dataclass_tree,
)
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes


# -------------------------
# File loading
# -------------------------
def load_calibration_file(file_path: str) -> CalibrationData:
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    if file_path.endswith((".h5", ".hdf5")):
        native = load_dict_from_hdf5(file_path)
        return dict_to_dataclass_tree(native, known=known_classes)

    raise ValueError(f"Unsupported file type: {file_path}")


def choose_file_dialog():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select calibration file",
        filetypes=[("Calibration", "*.pkl *.h5 *.hdf5"), ("All files", "*.*")]
    )


# -------------------------
# Data extraction
# -------------------------
def collect_points(calibration_data, reference):
    all_freqs = []
    all_px = []

    grouped_freqs = []
    grouped_mean = []
    grouped_std = []

    for freq_block in calibration_data.measured_freqs:
        px_values = []

        for point in freq_block.cali_meas_points:
            fs = point.fitting_results
            if not fs.is_success:
                continue

            if reference == "left":
                px = fs.left_peak_center_px
            elif reference == "right":
                px = fs.right_peak_center_px
            else:
                px = fs.inter_peak_distance

            px_values.append(px)
            all_px.append(px)
            all_freqs.append(freq_block.set_freq_ghz)

        if px_values:
            grouped_freqs.append(freq_block.set_freq_ghz)
            grouped_mean.append(np.mean(px_values))
            grouped_std.append(np.std(px_values))

    return (
        np.array(all_freqs),
        np.array(all_px),
        np.array(grouped_freqs),
        np.array(grouped_mean),
        np.array(grouped_std),
    )


# -------------------------
# Plotting
# -------------------------
def plot_reference(ax, calibration_data, calculator, reference, mode="poly"):
    all_f, all_px, g_f, g_mean, g_std = collect_points(calibration_data, reference)

    # scatter raw
    ax.scatter(all_f, all_px, s=10, alpha=0.3, label="raw")

    # grouped
    ax.errorbar(g_f, g_mean, yerr=g_std, fmt="o", label="mean ± std")

    # fit curve (px -> freq, so invert)
    px_fit = np.linspace(all_px.min(), all_px.max(), 400)

    if reference == "left":
        func = calculator.freq_left_peak if mode == "poly" else calculator.freq_left_peak_interp
        ylabel = "Left Peak (px)"
    elif reference == "right":
        func = calculator.freq_right_peak if mode == "poly" else calculator.freq_right_peak_interp
        ylabel = "Right Peak (px)"
    else:
        func = calculator.freq_peak_distance if mode == "poly" else calculator.freq_peak_distance_interp
        ylabel = "Distance (px)"

    freq_fit = func(px_fit)

    ax.plot(freq_fit, px_fit, "--", label="fit")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(reference)
    ax.grid(True)
    ax.legend()


# -------------------------
# Main
# -------------------------
def main():
    # CLI or dialog
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = choose_file_dialog()

    if not file_path:
        print("No file selected.")
        return

    print(f"Loading: {file_path}")

    calibration_data = load_calibration_file(file_path)

    config = calibration_config.get()

    calculator = get_calibration_calculator_from_data(calibration_data, poyfit_degree=config.degree)
    calculator.print_all_models()

    # Plot all
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, ref in zip(axes, ["left", "right", "distance"]):
        plot_reference(ax, calibration_data, calculator, ref, config.mode)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()