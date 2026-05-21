#!/usr/bin/env python3
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from scipy.optimize import curve_fit

from brillouin_system.calibration.calibration import (
    get_calibration_calculator_from_data,
    CalibrationData,
)
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    load_dict_from_hdf5,
    dict_to_dataclass_tree,
)


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

    # Convert GHz residuals to MHz
    residuals_mhz = (model_values - freq_points) * 1000.0

    return px_points, residuals_mhz


def normalize_within_pixel(px_points, centered=False):
    px_points = np.asarray(px_points, dtype=float)

    if centered:
        # Range: [-0.5, 0.5)
        return (px_points + 0.5) % 1.0 - 0.5

    # Range: [0, 1)
    return px_points % 1.0


def print_stats(name, residuals):
    print(f"mean {name}: {np.mean(residuals)} MHz")
    print(f"std {name}: {np.std(residuals)} MHz")


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
    print_stats("left", left_residuals)

    right_px, right_residuals = compute_residuals(
        p.right_px_points,
        p.right_freq_points,
        calculator.freq_right_peak,
    )
    print_stats("right", right_residuals)

    dist_px, dist_residuals = compute_residuals(
        p.dist_px_points,
        p.dist_freq_points,
        calculator.freq_peak_distance,
    )
    print_stats("dist", dist_residuals)

    # Set centered=True for range [-0.5, 0.5)
    centered = False

    left_px_norm = normalize_within_pixel(left_px, centered=centered)
    right_px_norm = normalize_within_pixel(right_px, centered=centered)
    # Fit sinusoids
    left_fit = fit_sinusoid(left_px_norm, left_residuals)
    right_fit = fit_sinusoid(right_px_norm, right_residuals)

    print(
        f"Left fit: amplitude={left_fit[0]:.3f} MHz, "
        f"phase={left_fit[1]:.3f} rad, "
        f"offset={left_fit[2]:.3f} MHz"
    )

    print(
        f"Right fit: amplitude={right_fit[0]:.3f} MHz, "
        f"phase={right_fit[1]:.3f} rad, "
        f"offset={right_fit[2]:.3f} MHz"
    )
    dist_px_norm = normalize_within_pixel(dist_px, centered=centered)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    xfit = np.linspace(
        -0.5 if centered else 0.0,
        0.5 if centered else 1.0,
        1000,
    )

    # Left
    axes[0].scatter(left_px_norm, left_residuals, s=2)
    axes[0].plot(
        xfit,
        sinusoid(xfit, *left_fit),
        linewidth=2,
    )
    axes[0].axhline(0.0, linestyle="--")
    axes[0].set_title("Left")
    axes[0].set_xlabel("Pixel phase")
    axes[0].set_ylabel("Model - Measured Frequency (MHz)")
    axes[0].grid(True)

    # Right
    axes[1].scatter(right_px_norm, right_residuals, s=2)
    axes[1].plot(
        xfit,
        sinusoid(xfit, *right_fit),
        linewidth=2,
    )
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_title("Right")
    axes[1].set_xlabel("Pixel phase")
    axes[1].set_ylabel("Model - Measured Frequency (MHz)")
    axes[1].grid(True)

    axes[2].scatter(dist_px_norm, dist_residuals, s=2)
    axes[2].axhline(0.0, linestyle="--")
    axes[2].set_title("Distance")
    axes[2].set_xlabel("Pixel phase")
    axes[2].set_ylabel("Model - Measured Frequency (MHz)")
    axes[2].grid(True)

    if centered:
        for ax in axes:
            ax.set_xlim(-0.5, 0.5)
    else:
        for ax in axes:
            ax.set_xlim(0.0, 1.0)

    fig.suptitle("Calibration Residuals Folded Into One Pixel", fontsize=14)
    plt.show()


def sinusoid(x, amplitude, phase, offset):
    return amplitude * np.sin(2 * np.pi * x + phase) + offset


def fit_sinusoid(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initial guesses
    amp0 = 0.5 * (np.max(y) - np.min(y))
    phase0 = 0.0
    offset0 = np.mean(y)

    popt, _ = curve_fit(
        sinusoid,
        x,
        y,
        p0=[amp0, phase0, offset0],
    )

    return popt


def main():
    app = QApplication(sys.argv)

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Open Calibration File",
        "",
        "Calibration Files (*.pkl *.h5 *.hdf5);;All Files (*)",
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