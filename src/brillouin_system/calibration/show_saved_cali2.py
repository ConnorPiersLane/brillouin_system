#!/usr/bin/env python3

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog

from brillouin_system.calibration.calibration import (
    get_calibration_calculator_from_data,
    CalibrationData,
)
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
# Data extraction (distance only)
# -------------------------
def collect_distance_points(calibration_data):
    all_freqs = []
    all_px = []

    for freq_block in calibration_data.measured_freqs:
        for point in freq_block.cali_meas_points:
            fs = point.fitting_results
            if not fs.is_success:
                continue

            px = fs.inter_peak_distance

            all_px.append(px)
            all_freqs.append(freq_block.set_freq_ghz)

    return np.array(all_freqs), np.array(all_px)


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

    degree = 1
    calculator = get_calibration_calculator_from_data(calibration_data, degree)

    # Get raw data
    freqs, px = collect_distance_points(calibration_data)

    # Plot (simple plt style)
    plt.figure(figsize=(6, 5))

    # Small dots
    plt.plot(freqs, px, linestyle='None', marker='o', markersize=2)

    # Optional: overlay fit curve
    px_fit = np.linspace(px.min(), px.max(), 400)
    freq_fit = calculator.freq_peak_distance(px_fit)
    plt.plot(freq_fit, px_fit, linestyle='--')

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Distance (px)")
    plt.title("Distance Calibration")
    plt.grid(True)
    plt.xlim(5.6, 6.2)
    plt.ylim(29, 33)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()