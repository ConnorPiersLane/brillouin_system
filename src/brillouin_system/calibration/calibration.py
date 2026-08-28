
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    resolve_fit_options,
)
from brillouin_system.spectrum_fitting.dho import DhoAxes
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
    is_dho_fit,
    is_psf_fit,
)

log = get_logger(__name__)

# The production line is deliberately minimal (cleaned 2026-08-20): a
# calibration stores raw frames + set frequencies, NOTHING else — fits happen
# in exactly one place, calibrate(). Fields removed from stored data:
#   CalibrationMeasurementPoint.fitting_results (the live acquisition fit —
#     display-only, was never read by calibrate())
#   MeasurementsPerFreq.state_mode (a full SystemState snapshot per frequency
#     block — never read by anything)
# Old files still load: the HDF5 reader drops unknown fields, pickle restores
# them as plain instance attributes.


@dataclass
class CalibrationMeasurementPoint:
    frame: np.ndarray
    microwave_freq: float


@dataclass
class MeasurementsPerFreq:
    set_freq_ghz: float
    cali_meas_points: list[CalibrationMeasurementPoint]


@dataclass
class CalibrationData:
    """The stored calibration: raw sideband frames per swept EOM frequency."""
    measured_freqs: list[MeasurementsPerFreq]


@dataclass
class CalibrationPolyfitParameters:

    degree: int = 1
    freq_left_peak: Optional[np.ndarray] = field(default=None)
    freq_right_peak: Optional[np.ndarray] = field(default=None)
    freq_peak_distance: Optional[np.ndarray] = field(default=None)
    calibration_width_left_peak: Optional[np.ndarray] = field(default=None)
    calibration_width_right_peak: Optional[np.ndarray] = field(default=None)
    # px -> GHz tracks of the two OUTER VIPA orders, filled when the
    # reference fit is four-peak (n_peaks = 4 in the [reference] config —
    # the standard since 2026-08-21). Every order gets its own track from
    # the same sideband frames and the same fitting pass as the inner pair.
    # None on two-peak calibrations and on data saved before the field
    # existed. No outer WIDTH tracks: the outer taus are provisional
    # (positions yes, width claims no — 2026-08-20).
    freq_outer_left_peak: Optional[np.ndarray] = field(default=None)
    freq_outer_right_peak: Optional[np.ndarray] = field(default=None)

    # The measured sideband points behind the polynomials (one entry per
    # fitted calibration frame, sorted by px) — kept for calibration plots
    # and residual diagnostics; never used to EVALUATE the calibration (the
    # np.interp mode was removed 2026-08-20, polynomials are the only map).
    left_px_points: Optional[np.ndarray] = field(default=None)
    left_freq_points: Optional[np.ndarray] = field(default=None)
    right_px_points: Optional[np.ndarray] = field(default=None)
    right_freq_points: Optional[np.ndarray] = field(default=None)
    dist_px_points: Optional[np.ndarray] = field(default=None)
    dist_freq_points: Optional[np.ndarray] = field(default=None)
    outer_left_px_points: Optional[np.ndarray] = field(default=None)
    outer_left_freq_points: Optional[np.ndarray] = field(default=None)
    outer_right_px_points: Optional[np.ndarray] = field(default=None)
    outer_right_freq_points: Optional[np.ndarray] = field(default=None)


@dataclass
class FourPeakShift:
    """The four per-order frequency estimates of one fit and their
    inverse-variance combination. Frequencies in GHz, ordered left to
    right on the detector: outer_left, left, right, outer_right."""
    freqs_ghz: tuple[float, float, float, float]
    weights: tuple[float, float, float, float]
    combined_ghz: float


@dataclass
class AnalyzedFreqShifts:
    """One fit converted to GHz through a calibration (CalibrationCalculator
    .analyze). The former SpectrumAnalyzer, removed 2026-08-20: the calculator
    already owns every primitive, so the conversion lives here."""
    freq_shift_left_peak_ghz: float | None
    freq_shift_right_peak_ghz: float | None
    freq_shift_peak_distance_ghz: float | None
    # Raw fitted width, as the peak lands on the detector: still broadened by
    # the instrument, whatever the lineshape. Unchanged meaning for every model.
    hwhm_left_peak_ghz: float | None
    hwhm_right_peak_ghz: float | None
    # Instrument HWHM from the calibration sidebands, at each peak's own pixel,
    # and the sample linewidth left after subtracting it. The linewidth pair is
    # None unless the fit is PSF-convolved and the calibration carries a width
    # model (see CalibrationCalculator.sample_linewidth_ghz).
    instrument_hwhm_left_peak_ghz: float | None = None
    instrument_hwhm_right_peak_ghz: float | None = None
    linewidth_left_peak_ghz: float | None = None
    linewidth_right_peak_ghz: float | None = None
    # Four-peak standard (2026-08-21): each outer order's shift through its
    # OWN calibration track, and the inverse-variance combination of all
    # four per-order estimates (Thompson photon-term weights — the
    # brightest orders dominate). None unless BOTH the fit and the
    # calibration are four-peak. The combination is the precision
    # observable (validated 2026-08-13: 1.84 MHz diff-sd vs 2.07 for the
    # distance); for ABSOLUTE nu_B the inner-pair distance stays the
    # anchor (outer-order medians spread ~14 MHz absolute).
    freq_shift_outer_left_peak_ghz: float | None = None
    freq_shift_outer_right_peak_ghz: float | None = None
    freq_shift_combined_ghz: float | None = None


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

    # --- Outer-order tracks (four-peak calibrations only) ---

    def has_outer_tracks(self) -> bool:
        """True when this calibration carries the outer-order tracks."""
        return (self.p.freq_outer_left_peak is not None
                and self.p.freq_outer_right_peak is not None
                and np.all(np.isfinite(np.asarray(self.p.freq_outer_left_peak, dtype=float)))
                and np.all(np.isfinite(np.asarray(self.p.freq_outer_right_peak, dtype=float))))

    def freq_outer_left_peak(self, px):
        """Frequency of the outer-left VIPA order [GHz] at pixel position px."""
        return np.polyval(self.p.freq_outer_left_peak, px)

    def dfreq_dpx_outer_left_peak(self, px):
        """Slope d(freq)/d(px) for the outer-left order [GHz/pixel]."""
        return np.polyval(np.polyder(self.p.freq_outer_left_peak, m=1), px)

    def freq_outer_right_peak(self, px):
        """Frequency of the outer-right VIPA order [GHz] at pixel position px."""
        return np.polyval(self.p.freq_outer_right_peak, px)

    def dfreq_dpx_outer_right_peak(self, px):
        """Slope d(freq)/d(px) for the outer-right order [GHz/pixel]."""
        return np.polyval(np.polyder(self.p.freq_outer_right_peak, m=1), px)

    def df_outer_left_peak(self, px, dpx):
        """Convert dpx to GHz using the outer-left order's local slope."""
        return self.dfreq_dpx_outer_left_peak(px) * dpx

    def df_outer_right_peak(self, px, dpx):
        """Convert dpx to GHz using the outer-right order's local slope."""
        return self.dfreq_dpx_outer_right_peak(px) * dpx

    def combined_shift(self, fitting: FittedSpectrum) -> FourPeakShift | None:
        """ONE frequency measurement from the position estimates of all four
        peaks, or None when the fit or the calibration is not four-peak.

        Each order's fitted centre maps to the Brillouin shift through its
        own track, giving four estimates of the same quantity; they are
        combined by inverse-variance weighting. The weights are the Thompson
        photon terms, which only need RELATIVE variances, so the gain and
        all shared constants cancel:

            var_i  ∝  s_i^2 / N_i  ∝  (w_i a_i)^2 / (amp_i w_i)  =  a_i^2 w_i / amp_i

        with w the fitted width [px], a the track's local dispersion [GHz/px]
        and amp the fitted amplitude (N ∝ amp*w, the exact peak area). The
        photon term dominates the per-peak budget, so richer weights (read
        noise, background) would move the combination negligibly while
        dragging in the camera gain.
        """
        if (not fitting.is_success
                or fitting.outer_left_peak_center_px is None
                or not self.has_outer_tracks()):
            return None

        peaks = [
            (fitting.outer_left_peak_center_px, fitting.outer_left_peak_width_px,
             fitting.outer_left_peak_amplitude,
             self.freq_outer_left_peak, self.dfreq_dpx_outer_left_peak),
            (fitting.left_peak_center_px, fitting.left_peak_width_px,
             fitting.left_peak_amplitude,
             self.freq_left_peak, self.dfreq_dpx_left_peak),
            (fitting.right_peak_center_px, fitting.right_peak_width_px,
             fitting.right_peak_amplitude,
             self.freq_right_peak, self.dfreq_dpx_right_peak),
            (fitting.outer_right_peak_center_px, fitting.outer_right_peak_width_px,
             fitting.outer_right_peak_amplitude,
             self.freq_outer_right_peak, self.dfreq_dpx_outer_right_peak),
        ]

        freqs, weights = [], []
        for cen, wid, amp, freq_of_px, slope_of_px in peaks:
            freqs.append(float(freq_of_px(cen)))
            a = float(slope_of_px(cen))
            var = a * a * float(wid) / max(float(amp), 1e-12)
            weights.append(1.0 / var)

        w = np.asarray(weights, dtype=float)
        f = np.asarray(freqs, dtype=float)
        combined = float(np.sum(w * f) / np.sum(w))

        return FourPeakShift(
            freqs_ghz=tuple(f.tolist()),
            weights=tuple((w / np.sum(w)).tolist()),
            combined_ghz=combined,
        )

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

    def instrument_hwhm_ghz(self, px_left, px_right) -> tuple[float | None, float | None]:
        """Instrument HWHM in GHz at each sample peak's own pixel.

        The EOM sidebands are spectrally sharp next to anything the spectrometer
        can resolve (kHz laser linewidth, a synthesizer-narrow tone), so the
        width fitted from a calibration frame IS the instrument response. The
        stored polynomial is that width vs pixel, so it is evaluated where the
        sample peak actually sits, not where the sidebands were.

        Returns (None, None) when the calibration carries no width model — data
        saved before it was stored.
        """
        def one(coeffs, width_ghz, px):
            if (coeffs is None
                    or not np.all(np.isfinite(np.asarray(coeffs, dtype=float)))
                    or px is None):
                return None
            return float(abs(width_ghz(px)))

        return (
            one(self.p.calibration_width_left_peak,
                self.calibration_width_left_peak_ghz, px_left),
            one(self.p.calibration_width_right_peak,
                self.calibration_width_right_peak_ghz, px_right),
        )

    def dho_axes(self) -> DhoAxes:
        """The per-peak inputs a 'dho_x_psf' sample fit needs from THIS
        calibration: the inner pair's px->GHz frequency tracks and the
        instrument-width polynomials (Lorentzian HWHM [px], folded into the
        DHO kernel at each peak's own position).

        Raises when the calibration carries no width model (data saved
        before it was stored, or a degenerate width fit) — a DHO without
        the instrument width is not fittable, and its center correction
        scales as Gamma^2, so guessing would land directly in the resonance.
        """
        p = self.p

        def checked(coeffs, name):
            if coeffs is None or not np.all(
                    np.isfinite(np.asarray(coeffs, dtype=float))):
                raise ValueError(
                    f"This calibration cannot drive a 'dho_x_psf' fit: "
                    f"'{name}' is missing or non-finite. The DHO needs the "
                    f"inner pair's frequency tracks and instrument-width "
                    f"polynomials from the scan's own calibration."
                )
            return np.asarray(coeffs, dtype=float)

        return DhoAxes(
            freq_left_poly=checked(p.freq_left_peak, "freq_left_peak"),
            freq_right_poly=checked(p.freq_right_peak, "freq_right_peak"),
            instrument_width_left_poly=checked(
                p.calibration_width_left_peak, "calibration_width_left_peak"),
            instrument_width_right_poly=checked(
                p.calibration_width_right_peak, "calibration_width_right_peak"),
        )

    def hwhm_ghz(self, fitting: FittedSpectrum) -> tuple[float | None, float | None]:
        """Raw fitted HWHM of a fit's two peaks in GHz — still instrument-broadened.

        This is the measured width of the peak as it lands on the detector. It
        is what the precision bound needs; for the sample's own linewidth see
        sample_linewidth_ghz. ONE exception: a DHO fit's width parameter is
        the ACOUSTIC width (its kernel already contains the instrument
        Lorentzian), so for those fits this is the material width already.
        """
        if not fitting.is_success:
            return None, None

        return (
            float(abs(self.df_left_peak(
                fitting.left_peak_center_px, fitting.left_peak_width_px))),
            float(abs(self.df_right_peak(
                fitting.right_peak_center_px, fitting.right_peak_width_px))),
        )

    def sample_linewidth_ghz(self, fitting: FittedSpectrum) -> tuple[float | None, float | None]:
        """Sample HWHM in GHz: fitted width minus the instrument width.

        Linear subtraction, because Lorentzian widths add under convolution, and
        evaluated at each peak's own pixel. The camera kernel is already out of
        both terms — the pixel-response model removes it from the sample fit,
        and the calibration was fitted with the same lineshape (the fitter
        refuses to mix families), so the two widths mean the same thing.

        Returns (None, None) unless that holds: only pixel-response fits are
        the validated width recipe, and only a calibration carrying a width
        model can supply the instrument term.

        A DHO fit ('dho_x_psf') needs NO subtraction: the instrument
        Lorentzian was folded into its kernel at fit time, so the fitted
        width IS the sample's acoustic HWHM — subtracting again would
        double-count the instrument.
        """
        if not fitting.is_success:
            return None, None
        if is_dho_fit(fitting.model):
            return self.hwhm_ghz(fitting)
        if not is_psf_fit(fitting.model):
            return None, None

        raw_l, raw_r = self.hwhm_ghz(fitting)
        inst_l, inst_r = self.instrument_hwhm_ghz(
            fitting.left_peak_center_px, fitting.right_peak_center_px)
        if inst_l is None or inst_r is None:
            return None, None

        return raw_l - inst_l, raw_r - inst_r

    def analyze(self, fitting: FittedSpectrum) -> AnalyzedFreqShifts:
        """Convert one fit's pixel-domain results to GHz."""
        if not fitting.is_success:
            return AnalyzedFreqShifts(
                freq_shift_left_peak_ghz=None,
                freq_shift_right_peak_ghz=None,
                freq_shift_peak_distance_ghz=None,
                hwhm_left_peak_ghz=None,
                hwhm_right_peak_ghz=None,
            )

        hwhm_left, hwhm_right = self.hwhm_ghz(fitting)
        inst_left, inst_right = self.instrument_hwhm_ghz(
            fitting.left_peak_center_px, fitting.right_peak_center_px)
        width_left, width_right = self.sample_linewidth_ghz(fitting)
        combined = self.combined_shift(fitting)

        return AnalyzedFreqShifts(
            freq_shift_left_peak_ghz=self.freq_left_peak(fitting.left_peak_center_px),
            freq_shift_right_peak_ghz=self.freq_right_peak(fitting.right_peak_center_px),
            freq_shift_peak_distance_ghz=self.freq_peak_distance(fitting.inter_peak_distance),
            hwhm_left_peak_ghz=hwhm_left,
            hwhm_right_peak_ghz=hwhm_right,
            instrument_hwhm_left_peak_ghz=inst_left,
            instrument_hwhm_right_peak_ghz=inst_right,
            linewidth_left_peak_ghz=width_left,
            linewidth_right_peak_ghz=width_right,
            freq_shift_outer_left_peak_ghz=(combined.freqs_ghz[0]
                                            if combined is not None else None),
            freq_shift_outer_right_peak_ghz=(combined.freqs_ghz[3]
                                             if combined is not None else None),
            freq_shift_combined_ghz=(combined.combined_ghz
                                     if combined is not None else None),
        )

    def print_all_models(self):
        """Print all available calibration models."""
        print(self.get_str_all_models())

    def get_str_all_models(self) -> str:
        """Return all available calibration models as a formatted string."""
        lines = []
        lines.append("==== All Calibration Models ====")
        lines.append(self._poly_to_line("Left Peak", self.p.freq_left_peak))
        lines.append(self._poly_to_line("Right Peak", self.p.freq_right_peak))
        lines.append(self._poly_to_line("Inter-Peak Distance", self.p.freq_peak_distance))
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


def get_calibration_calculator_from_data(calibration_data: CalibrationData, polyfit_degree) -> CalibrationCalculator:
    return CalibrationCalculator(calibrate(data=calibration_data, polyfit_degree=polyfit_degree))


def calibration_calculator_for_scan(
        calibration_data: CalibrationData | None,
        calibration_params: CalibrationPolyfitParameters | None,
        fitter: SpectrumFitter) -> CalibrationCalculator:
    """A scan's own calibration, re-fitted from its raw frames when possible.

    Takes only the scan's calibration information (AxialScan.calibration_data
    and .calibration_params) plus the fitter that will fit the samples.

    calibration_params was fitted at ACQUISITION time with whatever reference
    model was live then, so it silently pins the peak-centre convention of
    that model. Re-analysing samples with a different lineshape against it is
    the model-mixing trap (~0.27 px, -168 MHz split) that the fitter's guard
    catches between the two live configs but cannot see here. Re-fitting the
    stored frames with the current configs is what keeps the calibration and
    the samples on the same convention.

    Without the raw frames there is nothing to re-fit and no record of which
    model produced the stored polynomial, so a PSF-convolved re-analysis of
    such a scan is refused rather than quietly mixed.
    """
    if calibration_data is not None:
        degree = (calibration_params.degree
                  if calibration_params is not None
                  else calibration_config.get().degree)
        params = calibrate(data=calibration_data, polyfit_degree=degree,
                           fitter=fitter)
        log.info(f"[calibration] Re-fitted the scan's calibration from its raw "
                 f"frames (model={fitter.reference_config.fitting_model}, "
                 f"degree={degree}) — shifts may differ from the stored analysis.")
        return CalibrationCalculator(parameters=params)

    if resolve_fit_options(fitter.sample_config).model == "lorentzian_x_psf":
        raise ValueError(
            "The scan carries no raw calibration frames (calibration_data is "
            "None: recorded before they were stored, or with the old "
            "save_calibration_frames toggle off — removed 2026-08-24, frames "
            "always travel now), so its calibration cannot be "
            "re-fitted and there is no record of the model it was fitted "
            "with. A PSF-convolved sample fit against a calibration that is "
            "most likely lorentzian is the -168 MHz mixing trap. Analyse "
            "this scan with 'lorentzian' instead."
        )

    log.info("[calibration] No raw calibration frames stored — using the "
             "calibration polynomial as fitted at acquisition time.")
    return CalibrationCalculator(parameters=calibration_params)


def sort_xy(x, y):
    idx = np.argsort(x)
    return np.asarray(x)[idx], np.asarray(y)[idx]


def calibrate(data: CalibrationData, polyfit_degree,
              fitter: SpectrumFitter | None = None) -> CalibrationPolyfitParameters:
    """Fit a calibration from its raw frames.

    Pass the same fitter used for the samples when re-fitting a scan's own
    calibration: it carries that scan's row band, and the row band must not move
    between a calibration and its samples (~3-4 MHz per row). A fitter built
    here reads the configs as they are NOW, which is what a re-analysis wants —
    the model can only be changed by re-fitting.
    """
    degree = polyfit_degree
    sf = fitter if fitter is not None else SpectrumFitter()

    all_fits = []
    freqs_all = []

    for freq_block in data.measured_freqs:
        for point in freq_block.cali_meas_points:
            px, sline = sf.get_px_sline_from_image(point.frame)
            fs = sf.fit(px, sline, is_reference_mode=True)
            if fs.is_success:
                all_fits.append(fs)
                freqs_all.append(point.microwave_freq)

    if not all_fits:
        hint = ("" if int(sf.sline_config.n_peaks) != 4 else
                " n_peaks=4 is set (global fitting config) but no frame "
                "yielded a four-peak fit — this calibration was likely "
                "recorded with a two-peak ROI (or the reference thresholds "
                "miss the outer orders). Set n_peaks=2 for this data.")
        raise ValueError("No successful fits found in calibration data." + hint)

    freqs_all = np.asarray(freqs_all, dtype=float)
    left_px = np.asarray([fs.left_peak_center_px for fs in all_fits], dtype=float)
    right_px = np.asarray([fs.right_peak_center_px for fs in all_fits], dtype=float)
    inter_px = np.asarray([fs.inter_peak_distance for fs in all_fits], dtype=float)
    left_width = np.asarray([fs.left_peak_width_px for fs in all_fits], dtype=float)
    right_width = np.asarray([fs.right_peak_width_px for fs in all_fits], dtype=float)

    def safe_polyfit(x, y, deg):
        if len(x) <= deg:
            log.warning(f"[calibration] Not enough points for degree {deg} fit (got {len(x)} points).")
            return np.full(deg + 1, np.nan)
        return np.polyfit(x, y, deg)

    # The measured points travel with the parameters (one entry per fitted
    # frame, sorted by px) — for plots and residual diagnostics.
    left_px_sorted, left_freq_sorted = sort_xy(left_px, freqs_all)
    right_px_sorted, right_freq_sorted = sort_xy(right_px, freqs_all)
    dist_px_sorted, dist_freq_sorted = sort_xy(inter_px, freqs_all)

    params = CalibrationPolyfitParameters(
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

    # Four-peak calibration (the standard where the ROI allows it): the SAME
    # fits carry the outer-order sideband positions, so every order gets its
    # own track from the one fitting pass — nothing is refitted.
    if all(fs.outer_left_peak_center_px is not None for fs in all_fits):
        outer_left_px = np.asarray(
            [fs.outer_left_peak_center_px for fs in all_fits], dtype=float)
        outer_right_px = np.asarray(
            [fs.outer_right_peak_center_px for fs in all_fits], dtype=float)
        params.freq_outer_left_peak = safe_polyfit(outer_left_px, freqs_all, degree)
        params.freq_outer_right_peak = safe_polyfit(outer_right_px, freqs_all, degree)
        (params.outer_left_px_points,
         params.outer_left_freq_points) = sort_xy(outer_left_px, freqs_all)
        (params.outer_right_px_points,
         params.outer_right_freq_points) = sort_xy(outer_right_px, freqs_all)

    return params
