import io
from dataclasses import dataclass
from typing import List, Union, Optional, Callable

import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from brillouin_system.config.config import CalibrationConfig
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.my_dataclasses.state_mode import StateMode


@dataclass
class CalibrationMeasurementPoint:
    """
    Represents a single measurement point during calibration.
    """
    frame: np.ndarray  # Original frame, not background subtracted
    microwave_freq: float  # Actually measured frequency (GHz)
    fitting_results: FittedSpectrum


@dataclass
class MeasurementsPerFreq:
    """
    Contains all measurements taken for a given set frequency.
    """
    set_freq_ghz: float
    state_mode: StateMode
    cali_meas_points: List[CalibrationMeasurementPoint]


@dataclass
class CalibrationData:
    """
    Container for all calibration measurements.
    """
    measured_freqs: List[MeasurementsPerFreq]


@dataclass
class Calibration:
    """
    Quadratic calibration model: freq = a*x^2 + b*x + c
    """
    a: float
    b: float
    c: float

    def get_freq(self, x_px: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Calculate frequency from pixel positions using the quadratic model.

        Args:
            x_px: pixel positions (float, list, or np.ndarray)

        Returns:
            np.ndarray: corresponding frequencies in GHz
        """
        x_px_arr = np.asarray(x_px)
        return self.a * x_px_arr ** 2 + self.b * x_px_arr + self.c

    def get_dfreq(self, dx_px: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Calculate the derivative (df/dx)*dx_px from pixel positions using the quadratic model.

        Args:
            dx_px: delta pixel positions (float, list, or np.ndarray)

        Returns:
            np.ndarray: corresponding frequencies in GHz
        """
        dx_px_arr = np.asarray(dx_px)
        return 2 * self.a * dx_px_arr + self.b


@dataclass
class CalibrationResults:
    """
    Stores the calibration results for different references.
    """
    left_pixel: Calibration
    right_pixel: Calibration
    peak_distance: Calibration
    sigma_func_left_px: Optional[Callable[[float], float]] = None
    sigma_func_right_px: Optional[Callable[[float], float]] = None

    def get_calibration(self, config: CalibrationConfig) -> Calibration:
        if config.reference == "left":
            return self.left_pixel
        elif config.reference == "right":
            return self.right_pixel
        elif config.reference == "distance":
            return self.peak_distance
        else:
            raise ValueError(f"Unknown calibration reference: {config.reference}")

def create_sigma_func(positions: np.ndarray, sigmas: np.ndarray) -> callable:
    """
    Create an interpolating function for PSF sigma as a function of pixel position.

    Args:
        positions: Array of pixel positions where PSF sigma was measured.
        sigmas: Array of corresponding PSF sigma values.

    Returns:
        A function sigma_func(x) that interpolates sigma at position x.
    """
    sorted_indices = np.argsort(positions)
    sorted_positions = positions[sorted_indices]
    sorted_sigmas = sigmas[sorted_indices]

    def sigma_func(x):
        return np.interp(
            x,
            sorted_positions,
            sorted_sigmas,
            left=sorted_sigmas[0],
            right=sorted_sigmas[-1]
        )

    return sigma_func

def calibrate(data: CalibrationData) -> CalibrationResults:
    all_fits = []
    freqs = []

    for freq_block in data.measured_freqs:
        for point in freq_block.cali_meas_points:
            if point.fitting_results.is_success:
                all_fits.append(point.fitting_results)
                freqs.append(point.microwave_freq)

    freqs = np.array(freqs)
    left_px = np.array([fs.left_peak_center_px for fs in all_fits])
    right_px = np.array([fs.right_peak_center_px for fs in all_fits])
    inter_px = np.array([fs.inter_peak_distance for fs in all_fits])

    def fit(x: np.ndarray, y: np.ndarray) -> Calibration:
        if len(x) < 3:
            print("[Calibration] Not enough points for reliable fit.")
            return Calibration(a=np.nan, b=np.nan, c=np.nan)
        return Calibration(*np.polyfit(x, y, deg=2))

    # Build sigma lookup functions for left and right peaks separately
    left_sigma_positions = np.array([fs.left_peak_center_px for fs in all_fits])
    left_sigmas = np.array([fs.left_peak_width_px for fs in all_fits])

    right_sigma_positions = np.array([fs.right_peak_center_px for fs in all_fits])
    right_sigmas = np.array([fs.right_peak_width_px for fs in all_fits])

    sorted_left = np.argsort(left_sigma_positions)
    sorted_right = np.argsort(right_sigma_positions)
    left_sigma_func = create_sigma_func(left_sigma_positions, left_sigmas)
    right_sigma_func = create_sigma_func(right_sigma_positions, right_sigmas)

    return CalibrationResults(
        left_pixel=fit(left_px, freqs),
        right_pixel=fit(right_px, freqs),
        peak_distance=fit(inter_px, freqs),
        sigma_func_left_px=left_sigma_func,
        sigma_func_right_px=right_sigma_func
    )



def get_calibration_fig(calibration_data: CalibrationData, calibration_result: CalibrationResults, reference: str) -> Figure:
    """
    Creates a matplotlib figure displaying the calibration curve.

    Args:
        calibration_data: CalibrationData
        calibration_result: CalibrationResults containing fit parameters
        reference: 'left', 'right', or 'distance' to select calibration type

    Returns:
        matplotlib.figure.Figure object
    """
    assert reference in ["left", "right", "distance"], "Invalid reference type"

    calibration_data: CalibrationData = calibration_data

    # Helper functions for extracting the desired peak info
    def extract_left(fs: FittedSpectrum) -> float:
        return fs.left_peak_center_px

    def extract_right(fs: FittedSpectrum) -> float:
        return fs.right_peak_center_px

    def extract_distance(fs: FittedSpectrum) -> float:
        return fs.inter_peak_distance

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

    all_freqs = []
    all_pixels = []

    grouped_pixels = []
    grouped_freqs = []

    for freq_block in calibration_data.measured_freqs:
        valid_pixels = [
            extract(point.fitting_results)
            for point in freq_block.cali_meas_points
            if point.fitting_results.is_success
        ]
        if valid_pixels:
            grouped_pixels.append(valid_pixels)
            grouped_freqs.append(freq_block.set_freq_ghz)  # use the set freq for error bars

            all_freqs.extend([point.microwave_freq for point in freq_block.cali_meas_points
                              if point.fitting_results.is_success])
            all_pixels.extend(valid_pixels)

    if not all_freqs:
        raise ValueError("No successful calibration data to display.")

    fig, ax = plt.subplots()

    ax.scatter(all_freqs, all_pixels, color="blue", s=10, alpha=0.3, label="Measured Points")

    means = [np.mean(pixels) for pixels in grouped_pixels]
    stds = [np.std(pixels) for pixels in grouped_pixels]

    ax.errorbar(grouped_freqs, means, yerr=stds, fmt='o', color='orange',
                ecolor='gray', elinewidth=2, capsize=4, label='Mean ± StdDev')

    y_fit = np.linspace(min(all_pixels), max(all_pixels), 200)
    x_fit = calibration.get_freq(y_fit)
    ax.plot(x_fit, y_fit, 'r--', label="Fitted Curve")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Calibration Fit ({reference.capitalize()}: "
                 f"a={round(calibration.a, 2)} [GHz/px²], "
                 f"b={round(calibration.b, 2)} [GHz/px], "
                 f"c={round(calibration.c, 2)} [GHz])")
    print(f"Calibration Fit ({reference.capitalize()}: "
          f"a={round(calibration.a, 5)} [GHz/px²], "
          f"b={round(calibration.b, 4)} [GHz/px], "
          f"c={round(calibration.c, 4)} [GHz])")
    ax.grid(True)
    ax.legend()

    return fig


def render_calibration_to_pixmap(calibration_data: CalibrationData, calibration_results: CalibrationResults, reference: str) -> QPixmap:
    """
    Renders the calibration figure as a Qt pixmap.

    Args:
        calibration_results: CalibrationResults
        reference: 'left', 'right', or 'distance' to select calibration type

    Returns:
        QPixmap object
    """
    fig = get_calibration_fig(calibration_data, calibration_results, reference)

    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # Prevent display

    buf.seek(0)
    image = QImage.fromData(buf.getvalue(), format='PNG')
    return QPixmap.fromImage(image)


class CalibrationImageDialog(QDialog):
    """
    Simple dialog to display the calibration image.
    """
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
