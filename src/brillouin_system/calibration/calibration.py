import os
import io
from dataclasses import dataclass, field

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
    freq_peak_centroid: np.ndarray
    calibration_width_left_peak: np.ndarray
    calibration_width_right_peak: np.ndarray
    freq_DC_model: np.ndarray = field(default=None)
    """
    Linear regression coefficients [a, b, c] for the joint Distance–Centroid model:

        Freq ≈ a*D + b*C + c

    where
      D = (x2 - x1)   : inter-peak distance in pixels
      C = (x1 + x2)/2 : centroid in pixels

    Stored as a NumPy array of length 3: [a, b, c].
    """


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

    def compute_freq_shift(self, fitting: FittedSpectrum,) -> float | None:
        if not fitting.is_success:
            return None

        config = calibration_config.get()

        if config.reference == "left":
            return float(self.freq_left_peak(fitting.left_peak_center_px))
        elif config.reference == "right":
            return float(self.freq_right_peak(fitting.right_peak_center_px))
        elif config.reference == "distance":
            return float(self.freq_peak_distance(fitting.inter_peak_distance))
        elif config.reference == "centroid":
            return float(self.freq_peak_centroid((fitting.right_peak_center_px+fitting.left_peak_center_px)/2))
        elif config.reference == "dc":
            return self.freq_DC_model(
                D=fitting.inter_peak_distance,
                C=(fitting.right_peak_center_px+fitting.left_peak_center_px)/2,
            )
        else:
            return None

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

    def freq_peak_centroid(self, px):
        """Frequency Centroid: left and right peaks [GHz] at pixel position px: (x2+x1)/2"""
        return np.polyval(self.p.freq_peak_centroid, px)

    def freq_DC_model(self, D: float, C: float):
        """
        Predict frequency [GHz] from measured inter-peak distance and centroid
        using the stored Distance–Centroid model.

        Parameters
        ----------
        D : float or ndarray
            Inter-peak distance (x2 - x1) in pixels.
        C : float or ndarray
            Centroid (x1 + x2)/2 in pixels.

        Returns
        -------
        freq : float or ndarray
            Predicted frequency [GHz] from the fitted model.
        """
        a, b, c = self.p.freq_DC_model
        return DC_model(a=a, b=b, c=c, D=D, C=C)

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
    centroid_px = np.array([(fs.left_peak_center_px +  fs.right_peak_center_px)/2 for fs in all_fits])
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
        freq_peak_centroid=safe_polyfit(centroid_px, freqs, degree),
        calibration_width_left_peak=safe_polyfit(left_px, left_width, degree),
        calibration_width_right_peak=safe_polyfit(right_px, right_width, degree),
        freq_DC_model=np.array(fit_DC_model(inter_px, centroid_px, freqs)),
    )

    return params



def fit_DC_model(D, C, Freqs):
    """
    Fit the joint Distance–Centroid model:

        Freqs = a*D + b*C + c

    using ordinary least squares.

    Parameters
    ----------
    D : array_like
        Inter-peak distances (x2 - x1) in pixels.
    C : array_like
        Centroids (x1 + x2)/2 in pixels.
    Freqs : array_like
        Measured microwave frequencies in GHz.

    Returns
    -------
    a, b, c : floats
        Regression coefficients such that
        predicted Freq = a*D + b*C + c
    """
    D = np.asarray(D)
    C = np.asarray(C)
    Freqs = np.asarray(Freqs)

    # Design matrix with columns [D, C, 1]
    X = np.column_stack([D, C, np.ones_like(D)])

    # Solve least-squares problem
    coeffs, *_ = np.linalg.lstsq(X, Freqs, rcond=None)
    a, b, c = coeffs
    return a, b, c

def DC_model(a, b, c, D, C) -> float:
    """
    Evaluate the Distance–Centroid model:

        Freq = a*D + b*C + c

    Parameters
    ----------
    a, b, c : float
        Model coefficients from `fit_DC_model`.
    D : float or ndarray
        Inter-peak distance (x2 - x1) in pixels.
    C : float or ndarray
        Centroid (x1 + x2)/2 in pixels.

    Returns
    -------
    freq : float or ndarray
        Predicted frequency [GHz].
    """
    freq = a*D + b*C + c
    return freq


def get_calibration_fig(calibration_data: CalibrationData, calculator: CalibrationCalculator, reference: str) -> Figure:
    assert reference in ["left", "right", "distance", "centroid", "dc"], "Invalid reference type"

    if reference == "dc":
        return _get_calibration_fig_dc3d(calibration_data, calculator)

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

def _get_calibration_fig_dc3d(calibration_data: CalibrationData, calculator: CalibrationCalculator) -> Figure:
    """
    3D visualization of the Distance–Centroid model:
        Freq ≈ a*D + b*C + c
    Plots measured points (D, C, Freq) and the fitted plane.
    """
    # Gather data
    D_list, C_list, F_list = [], [], []
    for freq_block in calibration_data.measured_freqs:
        for point in freq_block.cali_meas_points:
            fs = point.fitting_results
            if getattr(fs, "is_success", False):
                D = fs.inter_peak_distance
                C = 0.5 * (fs.left_peak_center_px + fs.right_peak_center_px)
                F = point.microwave_freq
                if np.isfinite(D) and np.isfinite(C) and np.isfinite(F):
                    D_list.append(D)
                    C_list.append(C)
                    F_list.append(F)

    if not D_list:
        raise ValueError("No successful calibration data to display for DC model.")

    D = np.asarray(D_list)
    C = np.asarray(C_list)
    F = np.asarray(F_list)

    # Coefficients a, b, c
    if calculator.p.freq_DC_model is not None and len(calculator.p.freq_DC_model) == 3:
        a, b, c = calculator.p.freq_DC_model
    else:
        # Fallback: fit from the gathered points (useful if not stored)
        a, b, c = fit_DC_model(D, C, F)

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter of measured data
    ax.scatter(D, C, F, s=15, alpha=0.7, label="Measured Points")

    # Fitted plane
    D_lin = np.linspace(D.min(), D.max(), 30)
    C_lin = np.linspace(C.min(), C.max(), 30)
    Dg, Cg = np.meshgrid(D_lin, C_lin)
    Fg = a * Dg + b * Cg + c
    ax.plot_surface(Dg, Cg, Fg, alpha=0.25, edgecolor='none')

    # Labels & title
    ax.set_xlabel("Distance D (px)")
    ax.set_ylabel("Centroid C (px)")
    ax.set_zlabel("Frequency (GHz)")
    ax.set_title("Calibration Fit (Distance–Centroid, 3D)")
    ax.legend(loc="upper left")

    # Show plane equation in figure corner
    ax.text2D(0.02, 0.98, f"Freq ≈ {a:.3g}·D + {b:.3g}·C + {c:.3g}", transform=ax.transAxes, va="top")

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
