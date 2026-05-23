
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


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

    degree: int = 1
    freq_left_peak: Optional[np.ndarray] = field(default=None)
    freq_right_peak: Optional[np.ndarray] = field(default=None)
    freq_peak_distance: Optional[np.ndarray] = field(default=None)
    calibration_width_left_peak: Optional[np.ndarray] = field(default=None)
    calibration_width_right_peak: Optional[np.ndarray] = field(default=None)

    left_px_points: Optional[np.ndarray] = field(default=None) # must be increase (see np.interp)
    left_freq_points: Optional[np.ndarray] = field(default=None)
    right_px_points: Optional[np.ndarray] = field(default=None) # must be increase (see np.interp)
    right_freq_points: Optional[np.ndarray] = field(default=None)
    dist_px_points: Optional[np.ndarray] = field(default=None) # must be increase (see np.interp)
    dist_freq_points: Optional[np.ndarray] = field(default=None)


class CalibrationCalculator:
    """
    A utility class for evaluating calibration polynomial fits that map pixel positions to frequency-domain quantities.

    All methods take pixel coordinates (px) as input and return values in GHz.

    Parameters
    ----------
    parameters : CalibrationPolyfitParameters
        The polynomial fit coefficients for various calibration functions.
    """

    @staticmethod
    def interpolate_freq(px, px_points, freq_points):
        if px_points is None or freq_points is None:
            return None
        return np.interp(px, px_points, freq_points)

    def __init__(self, parameters: CalibrationPolyfitParameters):
        """Initialize the calculator with polynomial fit parameters."""
        self.p = parameters

    def compute_freq_shift(
            self,
            fitting: FittedSpectrum,
            reference: str = "distance",
            mode: str = "poly",
    ) -> float | None:
        """
        Compute frequency shift [GHz] from a fitted spectrum.

        Parameters
        ----------
        fitting : FittedSpectrum
            Result of the spectral fit.
        reference : str
            Which calibration reference to use:
            - "left"
            - "right"
            - "distance"
        mode : str
            Calibration mode:
            - "poly"
            - "interp"

        Returns
        -------
        float | None
            Frequency shift in GHz, or None if fit failed.

        Raises
        ------
        ValueError
            If reference or mode is invalid.
        """
        if not fitting.is_success:
            return None

        if mode not in {"poly", "interp"}:
            raise ValueError(f"Unknown mode '{mode}'. Use 'poly' or 'interp'.")

        if reference == "left":
            px_value = fitting.left_peak_center_px
            if mode == "poly":
                result = self.freq_left_peak(px_value)
            else:
                result = self.freq_left_peak_interp(px_value)

        elif reference == "right":
            px_value = fitting.right_peak_center_px
            if mode == "poly":
                result = self.freq_right_peak(px_value)
            else:
                result = self.freq_right_peak_interp(px_value)

        elif reference == "distance":
            px_value = fitting.inter_peak_distance
            if mode == "poly":
                result = self.freq_peak_distance(px_value)
            else:
                result = self.freq_peak_distance_interp(px_value)

        else:
            raise ValueError(
                f"Unknown reference '{reference}'. Use 'left', 'right', or 'distance'."
            )

        return None if result is None else float(result)

    def freq_left_peak(self, px):
        """Frequency of the left Brillouin peak [GHz] at pixel position px."""
        return np.polyval(self.p.freq_left_peak, px)

    def freq_left_peak_interp(self, px):
        """Frequency of the left Brillouin peak [GHz] at pixel position px."""
        return self.interpolate_freq(px, self.p.left_px_points, self.p.left_freq_points)

    def dfreq_dpx_left_peak(self, px):
        """Slope d(freq)/d(px) for left peak at pixel position px [GHz/pixel]."""
        coeffs = np.polyder(self.p.freq_left_peak, m=1)
        return np.polyval(coeffs, px)

    def freq_right_peak(self, px):
        """Frequency of the right Brillouin peak [GHz] at pixel position px."""
        return np.polyval(self.p.freq_right_peak, px)

    def freq_right_peak_interp(self, px):
        """Frequency of the left Brillouin peak [GHz] at pixel position px."""
        return self.interpolate_freq(px, self.p.right_px_points, self.p.right_freq_points)

    def dfreq_dpx_right_peak(self, px):
        """Slope d(freq)/d(px) for right peak at pixel position px [GHz/pixel]."""
        coeffs = np.polyder(self.p.freq_right_peak, m=1)
        return np.polyval(coeffs, px)

    def freq_peak_distance(self, px):
        """Frequency distance between left and right peaks [GHz] at pixel position px."""
        return np.polyval(self.p.freq_peak_distance, px)

    def freq_peak_distance_interp(self, px):
        """Frequency of the left Brillouin peak [GHz] at pixel position px."""
        return self.interpolate_freq(px, self.p.dist_px_points, self.p.dist_freq_points)


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

    def print_all_models(self):
        """Print all available calibration models."""
        print("==== All Calibration Models ====")
        self._print_poly("Left Peak", self.p.freq_left_peak)
        self._print_poly("Right Peak", self.p.freq_right_peak)
        self._print_poly("Inter-Peak Distance", self.p.freq_peak_distance)
        # self._print_poly("Centroid", self.p.freq_peak_centroid)
        # self._print_dc_model()
        print("================================")

    def get_str_all_models(self) -> str:
        """Return all available calibration models as a formatted string."""
        lines = []
        lines.append("==== All Calibration Models ====")
        lines.append(self._poly_to_line("Left Peak", self.p.freq_left_peak))
        lines.append(self._poly_to_line("Right Peak", self.p.freq_right_peak))
        lines.append(self._poly_to_line("Inter-Peak Distance", self.p.freq_peak_distance))
        # lines.append(self._poly_to_line("Centroid", self.p.freq_peak_centroid))
        # lines.append(self._dc_model_to_line())
        lines.append("================================")
        return "\n".join(lines)

    def _poly_to_line(self, name: str, coeffs: np.ndarray) -> str:
        eq = self._poly_to_str(coeffs)
        return f"{name}: f(x) ≈ {eq}  [GHz]"

    # --- Internal helpers ---
    @staticmethod
    def _poly_to_str(coeffs: np.ndarray) -> str:
        if coeffs is None or not np.all(np.isfinite(coeffs)):
            return "N/A"
        terms = []
        deg = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            power = deg - i
            if power == 0:
                terms.append(f"{c:.4g}")
            elif power == 1:
                terms.append(f"{c:.4g}·x")
            else:
                terms.append(f"{c:.4g}·x^{power}")
        return " + ".join(terms) if terms else "0"

    def _print_poly(self, name: str, coeffs: np.ndarray):
        eq = self._poly_to_str(coeffs)
        print(f"{name}: f(x) ≈ {eq}  [GHz]")


def get_calibration_calculator_from_data(calibration_data: CalibrationData, poyfit_degree) -> CalibrationCalculator:
    return CalibrationCalculator(calibrate(data=calibration_data, poyfit_degree=poyfit_degree))

def sort_xy(x, y):
    idx = np.argsort(x)
    return np.asarray(x)[idx], np.asarray(y)[idx]

sf = SpectrumFitter()

def calibrate(data: CalibrationData, poyfit_degree) -> CalibrationPolyfitParameters:
    degree = poyfit_degree

    all_fits = []
    freqs_all = []

    for freq_block in data.measured_freqs:
        for point in freq_block.cali_meas_points:
            # if point.fitting_results.is_success:
            #     all_fits.append(point.fitting_results)
            #     freqs_all.append(point.microwave_freq)
            px, sline = sf.get_px_sline_from_image(point.frame)
            fs = sf.fit(px, sline, is_reference_mode=True)
            if fs.is_success:
                all_fits.append(fs)
                freqs_all.append(point.microwave_freq)

    if not all_fits:
        raise ValueError("No successful fits found in calibration data.")

    freqs_all = np.asarray(freqs_all, dtype=float)
    left_px = np.asarray([fs.left_peak_center_px for fs in all_fits], dtype=float)
    right_px = np.asarray([fs.right_peak_center_px for fs in all_fits], dtype=float)
    inter_px = np.asarray([fs.inter_peak_distance for fs in all_fits], dtype=float)
    left_width = np.asarray([fs.left_peak_width_px for fs in all_fits], dtype=float)
    right_width = np.asarray([fs.right_peak_width_px for fs in all_fits], dtype=float)

    def safe_polyfit(x, y, deg):
        if len(x) <= deg:
            print(f"[Calibration Warning] Not enough points for degree {deg} fit (got {len(x)} points).")
            return np.full(deg + 1, np.nan)
        return np.polyfit(x, y, deg)

    left_peaks_mean = []
    right_peaks_mean = []
    distance_peaks_mean = []
    freqs_mean = []

    fits_by_freq = {}

    for fs, freq in zip(all_fits, freqs_all):
        fits_by_freq.setdefault(freq, []).append(fs)

    for freq, fits in fits_by_freq.items():
        freqs_mean.append(freq)
        left_peaks_mean.append(np.mean([fs.left_peak_center_px for fs in fits]))
        right_peaks_mean.append(np.mean([fs.right_peak_center_px for fs in fits]))
        distance_peaks_mean.append(np.mean([fs.inter_peak_distance for fs in fits]))

    left_px_sorted, left_freq_sorted = sort_xy(np.asarray(left_peaks_mean), np.asarray(freqs_mean))
    right_px_sorted, right_freq_sorted = sort_xy(np.asarray(right_peaks_mean), np.asarray(freqs_mean))
    dist_px_sorted, dist_freq_sorted = sort_xy(np.asarray(distance_peaks_mean), np.asarray(freqs_mean))

    return CalibrationPolyfitParameters(
        degree=degree,
        freq_left_peak=safe_polyfit(left_px, freqs_all, degree),
        freq_right_peak=safe_polyfit(right_px, freqs_all, degree),
        freq_peak_distance=safe_polyfit(inter_px, freqs_all, degree),
        calibration_width_left_peak=safe_polyfit(left_px, left_width, degree),
        calibration_width_right_peak=safe_polyfit(right_px, right_width, degree),
        left_px_points=left_px_sorted,
        left_freq_points=left_freq_sorted,
        right_px_points=right_px_sorted,
        right_freq_points=right_freq_sorted,
        dist_px_points=dist_px_sorted,
        dist_freq_points=dist_freq_sorted,
    )
