import io

import matplotlib
import numpy as np

# Use a non-interactive backend for safety in GUI/headless contexts.
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel

from brillouin_system.calibration.calibration import CalibrationCalculator


_VALID_REFERENCES = {"left", "right", "distance"}


def _validate_reference(reference: str) -> None:
    if reference not in _VALID_REFERENCES:
        raise ValueError(
            f"Invalid reference '{reference}'. Expected one of {_VALID_REFERENCES}."
        )


def _get_reference_label(reference: str) -> str:
    return {
        "left": "Left Peak Position (px)",
        "right": "Right Peak Position (px)",
        "distance": "Inter-Peak Distance (px)",
    }[reference]


def _calibration_points(calculator: CalibrationCalculator, reference: str):
    """The measured (px, GHz) sideband points stored with the parameters,
    plus per-frequency mean/std for the error bars."""
    p = calculator.p
    px, freqs = {
        "left": (p.left_px_points, p.left_freq_points),
        "right": (p.right_px_points, p.right_freq_points),
        "distance": (p.dist_px_points, p.dist_freq_points),
    }[reference]
    if px is None or freqs is None or len(np.atleast_1d(px)) == 0:
        raise ValueError("Calibration carries no measured points to display.")
    px = np.asarray(px, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    grouped_freqs = np.unique(freqs)
    grouped_means = np.array([px[freqs == f].mean() for f in grouped_freqs])
    grouped_stds = np.array([px[freqs == f].std() for f in grouped_freqs])
    return freqs, px, grouped_freqs, grouped_means, grouped_stds


def get_calibration_fig(
    calculator: CalibrationCalculator,
    reference: str,
) -> Figure:
    _validate_reference(reference)

    freq_func = {
        "left": calculator.freq_left_peak,
        "right": calculator.freq_right_peak,
        "distance": calculator.freq_peak_distance,
    }[reference]
    y_label = _get_reference_label(reference)

    (
        all_freqs,
        all_pixels,
        grouped_freqs,
        grouped_means,
        grouped_stds,
    ) = _calibration_points(calculator, reference)

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

    # Calibration polynomials map px -> GHz, so build the curve in px-space.
    px_fit = np.linspace(float(np.min(all_pixels)), float(np.max(all_pixels)), 400)
    freq_fit = freq_func(px_fit)
    ax.plot(freq_fit, px_fit, "--", label="Calibration (poly)")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Calibration Fit ({reference.capitalize()})")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    return fig


def render_calibration_to_pixmap(
    calculator: CalibrationCalculator,
    reference: str,
) -> QPixmap:
    fig = get_calibration_fig(calculator=calculator, reference=reference)

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
