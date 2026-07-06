#!/usr/bin/env python3
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox


from brillouin_system.calibration.calibration import get_calibration_calculator_from_data, CalibrationData
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree


def load_calibration_file(file_path: str) -> CalibrationData:
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    if file_path.endswith((".h5", ".hdf5")):
        native = load_dict_from_hdf5(file_path)


        return dict_to_dataclass_tree(native, known=known_classes)

    raise ValueError(f"Unsupported file type: {file_path}")

def compute_residuals(px_points, freq_points, model_func):
    px_points = np.asarray(px_points, dtype=float)
    freq_points = np.asarray(freq_points, dtype=float)
    model_values = np.asarray(model_func(px_points), dtype=float)
    residuals = (model_values - freq_points)*1000
    return px_points, residuals


def plot_residuals(calculator):
    p = calculator.p

    if p.left_px_points is None or p.left_freq_points is None:
        raise ValueError("Missing left calibration points.")
    if p.right_px_points is None or p.right_freq_points is None:
        raise ValueError("Missing right calibration points.")
    if p.dist_px_points is None or p.dist_freq_points is None:
        raise ValueError("Missing distance calibration points.")

    left_px, left_residuals = compute_residuals(
        p.left_px_points,
        p.left_freq_points,
        calculator.freq_left_peak,
    )
    print(f"mean left: {np.mean(left_residuals)}")
    print(f"std left {np.std(left_residuals)}")

    right_px, right_residuals = compute_residuals(
        p.right_px_points,
        p.right_freq_points,
        calculator.freq_right_peak,
    )
    print(f"mean right: {np.mean(right_residuals)}")
    print(f"std right {np.std(right_residuals)}")

    dist_px, dist_residuals = compute_residuals(
        p.dist_px_points,
        p.dist_freq_points,
        calculator.freq_peak_distance,
    )
    print(f"mean dist: {np.mean(dist_residuals)}")
    print(f"std dist {np.std(dist_residuals)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    axes[0].scatter(left_px, left_residuals, s=2)
    axes[0].axhline(0.0, linestyle="--")
    axes[0].set_title("Left")
    axes[0].set_xlabel("Pixel")
    axes[0].set_ylabel("Model - Measured Frequency (MHz)")
    axes[0].grid(True)

    axes[1].scatter(right_px, right_residuals, s=0.5)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_title("Right")
    axes[1].set_xlabel("Pixel")
    axes[1].set_ylabel("Model - Measured Frequency (GHz)")
    axes[1].grid(True)

    axes[2].scatter(dist_px, dist_residuals, s=2)
    axes[2].axhline(0.0, linestyle="--")
    axes[2].set_title("Distance")
    axes[2].set_xlabel("Pixel")
    axes[2].set_ylabel("Model - Measured Frequency (GHz)")
    axes[2].grid(True)

    fig.suptitle("Calibration Residuals", fontsize=14)
    plt.show()

    # fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    #
    # # Sort for clean line plotting
    # idx = np.argsort(dist_px)
    # x = dist_px[idx]
    # y = dist_residuals[idx]
    #
    # # Line plot
    # # ax.plot(x, y, linestyle='-', marker='.', markersize=1)
    #
    # # Horizontal zero line
    # ax.plot(x, y, linestyle='None', marker='.', markersize=3)
    #
    # # Vertical lines at integer pixel positions
    # px_min = int(np.floor(np.min(dist_px)))
    # px_max = int(np.ceil(np.max(dist_px)))
    #
    # for px in range(px_min, px_max + 1):
    #     ax.axvline(px, linestyle=":", linewidth=0.8, alpha=0.5)
    #
    # # Labels
    # ax.set_title("Distance Residuals")
    # ax.set_xlabel("Pixel")
    # ax.set_ylabel("Model - Measured Frequency (GHz)")
    #
    # plt.show()


def main():
    app = QApplication(sys.argv)

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Open Calibration File",
        "",
        "Calibration Files (*.pkl *.h5 *.hdf5);;All Files (*)"
    )

    if not file_path:
        return

    try:
        calibration_data = load_calibration_file(file_path)
        degree = calibration_config.get().degree
        calculator = get_calibration_calculator_from_data(calibration_data, degree)
        plot_residuals(calculator)

    except Exception as e:
        QMessageBox.critical(None, "Error", str(e))

    sys.exit(0)


if __name__ == "__main__":
    main()