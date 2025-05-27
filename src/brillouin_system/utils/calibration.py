import io
from dataclasses import dataclass

import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_template import FigureCanvas

from brillouin_system.config.config import CalibrationConfig
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

@dataclass
class CalibrationData:
    """
    len(freqs) == len(fitted_spectras)
    len(fitted_spectras[0]) == n_per_freq
    """
    n_per_freq: int
    freqs: list[float]
    fitted_spectras: list[list[FittedSpectrum]]


@dataclass
class Calibration:
    a: float
    b: float
    c: float

    def get_freq(self, x_px: float | list[float] | np.ndarray) -> np.ndarray:
        x_px_arr = np.asarray(x_px)
        return self.a * x_px_arr ** 2 + self.b * x_px_arr + self.c


@dataclass
class CalibrationResults:
    data: CalibrationData
    left_pixel: Calibration
    right_pixel: Calibration
    peak_distance: Calibration

    def get_calibration(self, config: CalibrationConfig) -> Calibration:
        if config.reference == "left":
            return self.left_pixel
        elif config.reference == "right":
            return self.right_pixel
        elif config.reference == "distance":
            return self.peak_distance
        else:
            raise ValueError(f"Unknown calibration reference: {config.reference}")


def calibrate(data: CalibrationData) -> CalibrationResults:
    # Flatten valid fits and corresponding frequencies
    all_fits = []
    freqs = []

    for freq, fs_list in zip(data.freqs, data.fitted_spectras):
        for fs in fs_list:
            if fs.is_success:
                all_fits.append(fs)
                freqs.append(freq)

    freqs = np.array(freqs)

    # Extract pixel values
    left_px = np.array([fs.left_peak_center_px for fs in all_fits])
    right_px = np.array([fs.right_peak_center_px for fs in all_fits])
    inter_px = np.array([fs.inter_peak_distance for fs in all_fits])

    # Fit: freq = a·x² + b·x + c
    def fit(x, y):
        if len(x) < 3:
            print("[Calibration] Not enough points for reliable fit.")
            return Calibration(a=np.nan, b=np.nan, c=np.nan)
        return Calibration(*np.polyfit(x, y, deg=2))

    return CalibrationResults(
        data=data,
        left_pixel=fit(left_px, freqs),
        right_pixel=fit(right_px, freqs),
        peak_distance=fit(inter_px, freqs),
    )


def get_calibration_fig(calibration_result: CalibrationResults,
                        reference: str) -> Figure:
    assert reference in ["left", "right", "distance"], "Invalid reference type"

    calibration_data: CalibrationData = calibration_result.data

    # Helper functions for extracting reference value
    def extract_left(fs): return fs.left_peak_center_px
    def extract_right(fs): return fs.right_peak_center_px
    def extract_distance(fs): return fs.inter_peak_distance

    if reference == "left":
        extract = extract_left
        calibration = calibration_result.left_pixel
        y_label = "Left Peak Position (px)"
    elif reference == "right":
        extract = extract_right
        calibration = calibration_result.right_pixel
        y_label = "Right Peak Position (px)"
    else:
        extract = extract_distance
        calibration = calibration_result.peak_distance
        y_label = "Inter-Peak Distance (px)"

    # Gather all valid (freq, pixel_value) pairs
    all_freqs = []
    all_pixels = []

    grouped_pixels = []
    grouped_freqs = []

    for freq, fs_list in zip(calibration_data.freqs, calibration_data.fitted_spectras):
        valid_pixels = [extract(fs) for fs in fs_list if fs.is_success]
        if valid_pixels:
            grouped_pixels.append(valid_pixels)
            grouped_freqs.append(freq)
            all_freqs.extend([freq] * len(valid_pixels))
            all_pixels.extend(valid_pixels)

    if not all_freqs:
        raise ValueError("No successful calibration data to display.")

    fig, ax = plt.subplots()

    # Scatter plot of all valid points
    ax.scatter(all_freqs, all_pixels, color="blue", s=10, alpha=0.3, label="Measured Points")

    # Plot mean ± std deviation as error bars
    means = [np.mean(pixels) for pixels in grouped_pixels]
    stds = [np.std(pixels) for pixels in grouped_pixels]

    ax.errorbar(grouped_freqs, means, yerr=stds, fmt='o', color='orange',
                ecolor='gray', elinewidth=2, capsize=4, label='Mean ± StdDev')

    # Fitted calibration curve
    y_fit = np.linspace(min(all_pixels), max(all_pixels), 200)
    x_fit = calibration.get_freq(y_fit)
    ax.plot(x_fit, y_fit, 'r--', label="Fitted Curve")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Calibration Fit ({reference.capitalize()}: "
                 f"a={round(calibration.a, ndigits=2)} [GHz/px^2], "
                 f"b={round(calibration.b, ndigits=2)} [GHz/px], "
                 f"c={round(calibration.c, ndigits=2)} [GHz])")
    print(f"Calibration Fit ({reference.capitalize()}: "
                 f"a={round(calibration.a, ndigits=5)} [GHz/px^2], "
                 f"b={round(calibration.b, ndigits=4)} [GHz/px], "
                 f"c={round(calibration.c, ndigits=4)} [GHz])")
    ax.grid(True)
    ax.legend()

    return fig


def render_calibration_to_pixmap(calibration_results, reference: str) -> QPixmap:
    fig = get_calibration_fig(calibration_results, reference)

    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # prevent it from trying to display

    buf.seek(0)
    image = QImage.fromData(buf.getvalue(), format='PNG')
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