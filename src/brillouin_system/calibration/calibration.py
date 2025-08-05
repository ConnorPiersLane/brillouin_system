import os
import io
from dataclasses import dataclass

import numpy as np
import matplotlib

from brillouin_system.calibration.config.calibration_config import calibration_config

# Use Agg backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel


from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.system_state import SystemState


# Headless environment detection
HEADLESS = os.environ.get("DISPLAY", "") == ""




@dataclass
class CalibrationMeasurementPoint:
    frame: np.ndarray
    microwave_freq: float
    fitting_results: FittedSpectrum


@dataclass
class MeasurementsPerFreq:
    set_freq_ghz: float
    state_mode: SystemState
    cali_meas_points: list[CalibrationMeasurementPoint]

"""
This is stored:
"""
@dataclass
class CalibrationData:
    measured_freqs: list[MeasurementsPerFreq]


@dataclass
class CalibrationPolyfitParameters:
    degree: int
    freq_left_peak: np.ndarray
    freq_right_peak: np.ndarray
    freq_peak_distance: np.ndarray
    calibration_width_left_peak: np.ndarray
    calibration_width_right_peak: np.ndarray


class CalibrationCalculator:
    """
    A utility class for evaluating calibration polynomial fits that map pixel positions to frequency-domain quantities.

    All methods take pixel coordinates (px) as input and return values in GHz.

    Parameters
    ----------
    parameters : CalibrationPolyfitParameters
        The polynomial fit coefficients for various calibration functions.
    """
    def __init__(self, parameters: CalibrationPolyfitParameters):
        """Initialize the calculator with polynomial fit parameters."""
        self.p = parameters

    def freq_left_peak(self, px):
        """Frequency of the left Brillouin peak [GHz] at pixel position px."""
        return np.polyval(self.p.freq_left_peak, px)

    def dfreq_dpx_left_peak(self, px):
        """Slope d(freq)/d(px) for left peak at pixel position px [GHz/pixel]."""
        coeffs = np.polyder(self.p.freq_left_peak, m=1)
        return np.polyval(coeffs, px)

    def freq_right_peak(self, px):
        """Frequency of the right Brillouin peak [GHz] at pixel position px."""
        return np.polyval(self.p.freq_right_peak, px)

    def dfreq_dpx_right_peak(self, px):
        """Slope d(freq)/d(px) for right peak at pixel position px [GHz/pixel]."""
        coeffs = np.polyder(self.p.freq_right_peak, m=1)
        return np.polyval(coeffs, px)

    def freq_peak_distance(self, px):
        """Frequency distance between left and right peaks [GHz] at pixel position px."""
        return np.polyval(self.p.freq_peak_distance, px)

    def dfreq_dpx_peak_distance(self, px):
        """Slope d(distance)/d(px) of peak separation in GHz/pixel at pixel position px."""
        coeffs = np.polyder(self.p.freq_peak_distance, m=1)
        return np.polyval(coeffs, px)

    def df_left_peak(self, px, dpx):
        """Convert dpx to GHz using local slope of left peak."""
        slope = self.dfreq_dpx_left_peak(px)
        return slope * dpx

    def df_right_peak(self, px, dpx):
        """Convert dpx to GHz using local slope of right peak."""
        slope = self.dfreq_dpx_right_peak(px)
        return slope * dpx

    def df_peak_distance(self, px, dpx):
        """Convert dpx to GHz using local slope of peak distance."""
        slope = self.dfreq_dpx_peak_distance(px)
        return slope * dpx

    def calibration_width_left_peak_dpx(self, px):
        """Ideal FWHM width of the left peak in pixels."""
        return np.polyval(self.p.calibration_width_left_peak, px)

    def calibration_width_right_peak_dpx(self, px):
        """Ideal FWHM width of the right peak in pixels."""
        return np.polyval(self.p.calibration_width_right_peak, px)

    def calibration_width_left_peak_ghz(self, px):
        """
        Convert the width (FWHM) of the left Brillouin peak from pixels to GHz.

        Parameters
        ----------
        px : float or ndarray
            Pixel position(s)

        Returns
        -------
        float or ndarray
            Width in GHz
        """
        dpx = self.calibration_width_left_peak_dpx(px)
        return self.df_left_peak(px, dpx)

    def calibration_width_right_peak_ghz(self, px):
        """
        Convert the width (FWHM) of the right Brillouin peak from pixels to GHz.

        Parameters
        ----------
        px : float or ndarray
            Pixel position(s)

        Returns
        -------
        float or ndarray
            Width in GHz
        """
        dpx = self.calibration_width_right_peak_dpx(px)
        return self.df_right_peak(px, dpx)



def get_calibration_calculator_from_data(calibration_data: CalibrationData) -> CalibrationCalculator:
    return CalibrationCalculator(calibrate(data=calibration_data))

def calibrate(data: CalibrationData) -> CalibrationPolyfitParameters:
    config = calibration_config.get()
    degree = config.degree

    all_fits = []
    freqs = []

    for freq_block in data.measured_freqs:
        for point in freq_block.cali_meas_points:
            if point.fitting_results.is_success:
                all_fits.append(point.fitting_results)
                freqs.append(point.microwave_freq)

    if not all_fits:
        raise ValueError("No successful fits found in calibration data.")

    freqs = np.array(freqs)
    left_px = np.array([fs.left_peak_center_px for fs in all_fits])
    right_px = np.array([fs.right_peak_center_px for fs in all_fits])
    inter_px = np.array([fs.inter_peak_distance for fs in all_fits])
    left_width = np.array([fs.left_peak_width_px for fs in all_fits])
    right_width = np.array([fs.right_peak_width_px for fs in all_fits])

    def safe_polyfit(x, y, deg):
        if len(x) <= deg:
            print(f"[Calibration Warning] Not enough points for degree {deg} fit (got {len(x)} points).")
            return np.full(deg + 1, np.nan)
        return np.polyfit(x, y, deg)

    params = CalibrationPolyfitParameters(
        degree=degree,
        freq_left_peak=safe_polyfit(left_px, freqs, degree),
        freq_right_peak=safe_polyfit(right_px, freqs, degree),
        freq_peak_distance=safe_polyfit(inter_px, freqs, degree),
        calibration_width_left_peak=safe_polyfit(left_px, left_width, degree),
        calibration_width_right_peak=safe_polyfit(right_px, right_width, degree),
    )

    return params



def get_calibration_fig(calibration_data: CalibrationData, calculator: CalibrationCalculator, reference: str) -> Figure:
    assert reference in ["left", "right", "distance"], "Invalid reference type"

    def extract(fs: FittedSpectrum) -> float:
        return {
            "left": fs.left_peak_center_px,
            "right": fs.right_peak_center_px,
            "distance": fs.inter_peak_distance
        }[reference]

    func_map = {
        "left": (calculator.freq_left_peak, "Left Peak Position (px)"),
        "right": (calculator.freq_right_peak, "Right Peak Position (px)"),
        "distance": (calculator.freq_peak_distance, "Inter-Peak Distance (px)"),
    }
    func, y_label = func_map[reference]

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
            grouped_freqs.append(freq_block.set_freq_ghz)
            all_freqs.extend([point.microwave_freq for point in freq_block.cali_meas_points if point.fitting_results.is_success])
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
    x_fit = func(y_fit)
    ax.plot(x_fit, y_fit, 'r--', label="Fitted Curve")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Calibration Fit ({reference.capitalize()})")
    ax.grid(True)
    ax.legend()

    return fig


def render_calibration_to_pixmap(calibration_data: CalibrationData, calculator: CalibrationCalculator, reference: str) -> QPixmap:
    fig = get_calibration_fig(calibration_data, calculator, reference)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)

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
