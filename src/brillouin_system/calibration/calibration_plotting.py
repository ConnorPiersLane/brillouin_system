import io
import os
from typing import Callable

import matplotlib
import numpy as np

# Use a non-interactive backend for safety in GUI/headless contexts.
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel

from brillouin_system.calibration.calibration import CalibrationData, CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum


HEADLESS = os.environ.get("DISPLAY", "") == ""


_VALID_REFERENCES = {"left", "right", "distance"}
_VALID_MODES = {"poly", "interp"}


def _validate_reference(reference: str) -> None:
    if reference not in _VALID_REFERENCES:
        raise ValueError(
            f"Invalid reference '{reference}'. Expected one of {_VALID_REFERENCES}."
        )


def _validate_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Expected one of {_VALID_MODES}."
        )


def _extract_reference_value(fs: FittedSpectrum, reference: str) -> float:
    return {
        "left": fs.left_peak_center_px,
        "right": fs.right_peak_center_px,
        "distance": fs.inter_peak_distance,
    }[reference]


def _get_reference_label(reference: str) -> str:
    return {
        "left": "Left Peak Position (px)",
        "right": "Right Peak Position (px)",
        "distance": "Inter-Peak Distance (px)",
    }[reference]


def _get_calibration_function(
    calculator: CalibrationCalculator,
    reference: str,
    mode: str,
) -> Callable[[np.ndarray], np.ndarray]:
    if reference == "left":
        return calculator.freq_left_peak if mode == "poly" else calculator.freq_left_peak_interp
    if reference == "right":
        return calculator.freq_right_peak if mode == "poly" else calculator.freq_right_peak_interp
    return calculator.freq_peak_distance if mode == "poly" else calculator.freq_peak_distance_interp


def _collect_calibration_points(
    calibration_data: CalibrationData,
    reference: str,
):
    all_freqs: list[float] = []
    all_pixels: list[float] = []
    grouped_freqs: list[float] = []
    grouped_means: list[float] = []
    grouped_stds: list[float] = []

    for freq_block in calibration_data.measured_freqs:
        valid_pixels = [
            _extract_reference_value(point.fitting_results, reference)
            for point in freq_block.cali_meas_points
            if point.fitting_results.is_success
        ]

        if not valid_pixels:
            continue

        # Raw points
        all_pixels.extend(valid_pixels)
        all_freqs.extend([freq_block.set_freq_ghz] * len(valid_pixels))

        # Grouped summary
        grouped_freqs.append(freq_block.set_freq_ghz)
        grouped_means.append(float(np.mean(valid_pixels)))
        grouped_stds.append(float(np.std(valid_pixels)))

    if not all_freqs:
        raise ValueError("No successful calibration data to display.")

    return (
        np.asarray(all_freqs, dtype=float),
        np.asarray(all_pixels, dtype=float),
        np.asarray(grouped_freqs, dtype=float),
        np.asarray(grouped_means, dtype=float),
        np.asarray(grouped_stds, dtype=float),
    )


def get_calibration_fig(
    calibration_data: CalibrationData,
    calculator: CalibrationCalculator,
    reference: str,
    mode: str = "poly",
) -> Figure:
    _validate_reference(reference)
    _validate_mode(mode)

    freq_func = _get_calibration_function(calculator, reference, mode)
    y_label = _get_reference_label(reference)

    (
        all_freqs,
        all_pixels,
        grouped_freqs,
        grouped_means,
        grouped_stds,
    ) = _collect_calibration_points(calibration_data, reference)

    fig, ax = plt.subplots()

    ax.scatter(
        all_freqs,
        all_pixels,
        s=10,
        alpha=0.3,
        label="Measured Points",
    )

    ax.errorbar(
        grouped_freqs,
        grouped_means,
        yerr=grouped_stds,
        fmt="o",
        ecolor="gray",
        elinewidth=1.5,
        capsize=4,
        label="Mean ± StdDev",
    )

    # Calibration functions map px -> GHz, so build the curve in px-space.
    px_fit = np.linspace(float(np.min(all_pixels)), float(np.max(all_pixels)), 400)
    freq_fit = freq_func(px_fit)
    ax.plot(freq_fit, px_fit, "--", label=f"Calibration ({mode})")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Calibration Fit ({reference.capitalize()}, {mode})")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    return fig


def render_calibration_to_pixmap(
    calibration_data: CalibrationData,
    calculator: CalibrationCalculator,
    reference: str,
    mode: str = "poly",
) -> QPixmap:
    fig = get_calibration_fig(
        calibration_data=calibration_data,
        calculator=calculator,
        reference=reference,
        mode=mode,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)
    image = QImage.fromData(buf.getvalue(), format="PNG")
    return QPixmap.fromImage(image)


class CalibrationImageDialog(QDialog):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Plot")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()

        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        layout.addWidget(label)

        self.setLayout(layout)