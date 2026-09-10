
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.dho import DhoAxes
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    SpectrumFitter,
    is_dho_fit,
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
    # Degree of the outer-order FREQUENCY tracks (2026-09-10; None on data
    # saved before the field existed = same as `degree`). The outer tracks
    # bend more than a parabola: degree 3 removes a 2-3 MHz systematic in
    # the water band. Width tracks always use `degree`.
    outer_degree: Optional[int] = field(default=None)
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
    # existed.
    freq_outer_left_peak: Optional[np.ndarray] = field(default=None)
    freq_outer_right_peak: Optional[np.ndarray] = field(default=None)
    # Outer-order instrument-width tracks (2026-09-02): same meaning as the
    # inner pair's — fitted sideband width vs pixel, from the same four-peak
    # fitting pass. They make the outer orders full width observables
    # (instrument HWHM + sample linewidth), with one caveat: the outer PSF
    # taus are less validated than the inner pair's (marked provisional
    # 2026-08-20), so outer widths inherit that systematic.
    calibration_width_outer_left_peak: Optional[np.ndarray] = field(default=None)
    calibration_width_outer_right_peak: Optional[np.ndarray] = field(default=None)
    # Outer-pair DISTANCE track (2026-09-10): outer_right - outer_left [px]
    # vs EOM frequency, the outer counterpart of freq_peak_distance. It makes
    # the outer pair a second distance observable (immune to the common-mode
    # chip walk like the inner one). Four-peak calibrations only.
    freq_outer_peak_distance: Optional[np.ndarray] = field(default=None)

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
    outer_dist_px_points: Optional[np.ndarray] = field(default=None)
    outer_dist_freq_points: Optional[np.ndarray] = field(default=None)


@dataclass
class FourPeakShift:
    """The four per-order frequency estimates of one fit and their
    inverse-variance combination. Frequencies in GHz, ordered left to
    right on the detector: outer_left, left, right, outer_right."""
    freqs_ghz: tuple[float, float, float, float]
    weights: tuple[float, float, float, float]
    combined_ghz: float


@dataclass
class WeightedDistance:
    """The inner-pair and outer-pair distance readings of one fit and their
    PHOTON-WEIGHTED average (2026-09-10, user rule): each pair is weighted
    by the photon number of its two peaks (peak areas in counts; the camera
    gain cancels). GHz throughout, weights normalised to one."""
    inner_ghz: float
    outer_ghz: float
    inner_weight: float
    outer_weight: float
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
    # Outer-pair distance + photon-weighted distance (2026-09-10): the
    # outer pair read through its OWN distance track, and the average of the
    # inner and outer distances weighted by each pair's photon numbers (user
    # rule - the candidate for the reported shift, under test; the inner
    # distance stays the reported value until it validates). None unless
    # BOTH the fit and the calibration are four-peak.
    freq_shift_outer_distance_ghz: float | None = None
    freq_shift_weighted_distance_ghz: float | None = None
    weighted_distance_inner_weight: float | None = None
    # Outer-order widths (2026-09-02): same three-layer story as the inner
    # pair — raw fitted HWHM, the instrument HWHM at that peak's own pixel
    # (from the outer width tracks), and the sample linewidth left after
    # subtraction. None unless the fit is four-peak AND the calibration
    # carries the outer tracks (+ outer width model for the last two).
    hwhm_outer_left_peak_ghz: float | None = None
    hwhm_outer_right_peak_ghz: float | None = None
    instrument_hwhm_outer_left_peak_ghz: float | None = None
    instrument_hwhm_outer_right_peak_ghz: float | None = None
    linewidth_outer_left_peak_ghz: float | None = None
    linewidth_outer_right_peak_ghz: float | None = None


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

    def has_outer_distance_track(self) -> bool:
        """True when this calibration carries the outer-pair distance track."""
        c = self.p.freq_outer_peak_distance
        return c is not None and bool(np.all(np.isfinite(np.asarray(c, dtype=float))))

    def freq_outer_peak_distance(self, px):
        """Brillouin shift [GHz] read from the OUTER pair distance px."""
        return np.polyval(self.p.freq_outer_peak_distance, px)

    def dfreq_dpx_outer_peak_distance(self, px):
        """Slope d(freq)/d(px) of the outer-pair distance track [GHz/pixel]."""
        return np.polyval(np.polyder(self.p.freq_outer_peak_distance, m=1), px)

    def df_outer_peak_distance(self, px, dpx):
        """Convert an outer-distance dpx to GHz using the local slope."""
        return self.dfreq_dpx_outer_peak_distance(px) * dpx

    def outer_distance_shift(self, fitting: FittedSpectrum) -> float | None:
        """The shift from the OUTER pair distance through its own track, or
        None when the fit or the calibration is not four-peak."""
        if (not fitting.is_success
                or fitting.outer_inter_peak_distance is None
                or not self.has_outer_distance_track()):
            return None
        return float(self.freq_outer_peak_distance(fitting.outer_inter_peak_distance))

    def weighted_distance(self, fitting: FittedSpectrum) -> WeightedDistance | None:
        """Inner and outer pair distances averaged with PHOTON-NUMBER weights
        (user rule 2026-09-10): w_inner = N_L + N_R, w_outer = N_OL + N_OR,
        with N = pi * amp * width (the exact peak area in counts, the same
        number PixelCountsAndPhotons reports; gain and pi cancel in the
        ratio). This is deliberately NOT the inverse-variance rule of
        combined_shift - the user asked for the plain photon weights. None
        when the fit or the calibration is not four-peak."""
        outer = self.outer_distance_shift(fitting)
        if outer is None:
            return None
        inner = float(self.freq_peak_distance(fitting.inter_peak_distance))

        def area(amp, wid):
            return abs(float(amp) * float(wid))

        w_in = (area(fitting.left_peak_amplitude, fitting.left_peak_width_px)
                + area(fitting.right_peak_amplitude, fitting.right_peak_width_px))
        w_out = (area(fitting.outer_left_peak_amplitude, fitting.outer_left_peak_width_px)
                 + area(fitting.outer_right_peak_amplitude, fitting.outer_right_peak_width_px))
        total = w_in + w_out
        if not np.isfinite(total) or total <= 0.0:
            return None
        w_in, w_out = w_in / total, w_out / total
        return WeightedDistance(
            inner_ghz=inner, outer_ghz=outer,
            inner_weight=float(w_in), outer_weight=float(w_out),
            combined_ghz=float(w_in * inner + w_out * outer),
        )

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

    def calibration_width_outer_left_peak_dpx(self, px):
        """Ideal HWHM width of the outer-left sideband in pixels."""
        return np.polyval(self.p.calibration_width_outer_left_peak, px)

    def calibration_width_outer_right_peak_dpx(self, px):
        """Ideal HWHM width of the outer-right sideband in pixels."""
        return np.polyval(self.p.calibration_width_outer_right_peak, px)

    def calibration_width_outer_left_peak_ghz(self, px):
        """Width of the outer-left sideband in GHz at pixel px."""
        dpx = self.calibration_width_outer_left_peak_dpx(px)
        return self.df_outer_left_peak(px, dpx)

    def calibration_width_outer_right_peak_ghz(self, px):
        """Width of the outer-right sideband in GHz at pixel px."""
        dpx = self.calibration_width_outer_right_peak_dpx(px)
        return self.df_outer_right_peak(px, dpx)

    def instrument_hwhm_outer_ghz(self, px_outer_left, px_outer_right
                                  ) -> tuple[float | None, float | None]:
        """Instrument HWHM in GHz at each OUTER peak's own pixel — the
        outer-order counterpart of instrument_hwhm_ghz. (None, None) when
        the calibration carries no outer width model (two-peak calibrations
        and data saved before 2026-09-02)."""
        def one(coeffs, width_ghz, px):
            if (coeffs is None
                    or not np.all(np.isfinite(np.asarray(coeffs, dtype=float)))
                    or px is None):
                return None
            return float(abs(width_ghz(px)))

        return (
            one(self.p.calibration_width_outer_left_peak,
                self.calibration_width_outer_left_peak_ghz, px_outer_left),
            one(self.p.calibration_width_outer_right_peak,
                self.calibration_width_outer_right_peak_ghz, px_outer_right),
        )

    def hwhm_outer_ghz(self, fitting: FittedSpectrum
                       ) -> tuple[float | None, float | None]:
        """Raw fitted HWHM of the OUTER orders in GHz — still
        instrument-broadened, like hwhm_ghz for the inner pair. (None, None)
        unless the fit is four-peak and the calibration carries the outer
        frequency tracks (needed for the px -> GHz slope)."""
        if (not fitting.is_success
                or fitting.outer_left_peak_center_px is None
                or not self.has_outer_tracks()):
            return None, None

        return (
            float(abs(self.df_outer_left_peak(
                fitting.outer_left_peak_center_px,
                fitting.outer_left_peak_width_px))),
            float(abs(self.df_outer_right_peak(
                fitting.outer_right_peak_center_px,
                fitting.outer_right_peak_width_px))),
        )

    def sample_linewidth_outer_ghz(self, fitting: FittedSpectrum
                                   ) -> tuple[float | None, float | None]:
        """Sample HWHM from the OUTER orders: fitted width minus the outer
        instrument width, at each outer peak's own pixel — the outer-order
        counterpart of sample_linewidth_ghz (same rules: pixel-response fits
        only, linear Lorentzian subtraction). A four-peak DHO fit
        (2026-09-05) needs NO subtraction, exactly like the inner pair:
        the outer instrument widths were folded into its kernels at fit
        time, so the fitted widths ARE acoustic already."""
        raw_l, raw_r = self.hwhm_outer_ghz(fitting)
        if raw_l is None:
            return None, None
        if is_dho_fit(fitting.model):
            return raw_l, raw_r
        # a plain-Lorentzian fit carries no instrument model: no linewidth
        return None, None

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
        """Instrument (EOM sideband) Lorentzian HWHM of the left peak in pixels, at px."""
        return np.polyval(self.p.calibration_width_left_peak, px)

    def calibration_width_right_peak_dpx(self, px):
        """Instrument (EOM sideband) Lorentzian HWHM of the right peak in pixels, at px."""
        return np.polyval(self.p.calibration_width_right_peak, px)

    def calibration_width_left_peak_ghz(self, px):
        """
        Instrument Lorentzian HWHM of the left peak in GHz, at px.

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
        Instrument Lorentzian HWHM of the right peak in GHz, at px.

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
        """The frequency tracks a 'dho_x_psf' sample fit needs from THIS
        calibration: the inner pair's px->GHz polynomials, plus the outer
        orders' on a four-peak calibration. The measured kernels and the
        envelope are added by analysis.fit_axial_scan.dho_axes_for_fit."""
        p = self.p

        def checked(coeffs, name):
            if coeffs is None or not np.all(
                    np.isfinite(np.asarray(coeffs, dtype=float))):
                raise ValueError(
                    f"This calibration cannot drive a 'dho_x_psf' fit: "
                    f"'{name}' is missing or non-finite. The DHO needs the "
                    f"inner pair's frequency tracks from the scan's own "
                    f"calibration."
                )
            return np.asarray(coeffs, dtype=float)

        def optional(coeffs):
            # outer tracks: present only on four-peak calibrations — a
            # missing/degenerate one downgrades to inner-only DHO
            # (has_outer False) instead of raising.
            if coeffs is None or not np.all(
                    np.isfinite(np.asarray(coeffs, dtype=float))):
                return None
            return np.asarray(coeffs, dtype=float)

        return DhoAxes(
            freq_left_poly=checked(p.freq_left_peak, "freq_left_peak"),
            freq_right_poly=checked(p.freq_right_peak, "freq_right_peak"),
            freq_outer_left_poly=optional(p.freq_outer_left_peak),
            freq_outer_right_poly=optional(p.freq_outer_right_peak),
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
        # a plain-Lorentzian fit carries no instrument model: no linewidth
        return None, None

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
        hwhm_ol, hwhm_or = self.hwhm_outer_ghz(fitting)
        inst_ol, inst_or = (self.instrument_hwhm_outer_ghz(
            fitting.outer_left_peak_center_px,
            fitting.outer_right_peak_center_px)
            if fitting.outer_left_peak_center_px is not None
            else (None, None))
        width_ol, width_or = self.sample_linewidth_outer_ghz(fitting)
        weighted = self.weighted_distance(fitting)

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
            freq_shift_outer_distance_ghz=(weighted.outer_ghz
                                           if weighted is not None else None),
            freq_shift_weighted_distance_ghz=(weighted.combined_ghz
                                              if weighted is not None else None),
            weighted_distance_inner_weight=(weighted.inner_weight
                                            if weighted is not None else None),
            hwhm_outer_left_peak_ghz=hwhm_ol,
            hwhm_outer_right_peak_ghz=hwhm_or,
            instrument_hwhm_outer_left_peak_ghz=inst_ol,
            instrument_hwhm_outer_right_peak_ghz=inst_or,
            linewidth_outer_left_peak_ghz=width_ol,
            linewidth_outer_right_peak_ghz=width_or,
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
        if self.p.freq_outer_left_peak is not None:
            lines.append(f"(inner degree {self.p.degree}, outer degree "
                         f"{self.p.outer_degree if self.p.outer_degree is not None else self.p.degree})")
            lines.append(self._poly_to_line("Outer-Left Peak", self.p.freq_outer_left_peak))
            lines.append(self._poly_to_line("Outer-Right Peak", self.p.freq_outer_right_peak))
            lines.append(self._poly_to_line("Outer-Pair Distance", self.p.freq_outer_peak_distance))
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
                 f"frames (template chain, degree={degree}) — shifts may "
                 f"differ from the stored analysis.")
        return CalibrationCalculator(parameters=params)

    log.info("[calibration] No raw calibration frames stored — using the "
             "calibration polynomial as fitted at acquisition time.")
    return CalibrationCalculator(parameters=calibration_params)


def sort_xy(x, y):
    idx = np.argsort(x)
    return np.asarray(x)[idx], np.asarray(y)[idx]


def resolve_outer_degree(outer_degree: int | None) -> int:
    """The degree of the outer-order frequency tracks: the argument, else
    the live calibration config (outer_degree, 3 since 2026-09-10)."""
    if outer_degree is not None:
        return int(outer_degree)
    return int(getattr(calibration_config.get(), "outer_degree", 3))


def calibrate(data: CalibrationData, polyfit_degree,
              fitter: SpectrumFitter | None = None,
              outer_degree: int | None = None) -> CalibrationPolyfitParameters:
    """Fit a calibration from its raw frames.

    Pass the same fitter used for the samples when re-fitting a scan's own
    calibration: it carries that scan's row band, and the row band must not move
    between a calibration and its samples (~3-4 MHz per row). A fitter built
    here reads the configs as they are NOW, which is what a re-analysis wants —
    the model can only be changed by re-fitting.

    outer_degree: degree of the outer-order frequency tracks (outer_left,
    outer_right, outer distance); None = the live calibration config. The
    inner tracks and every width track use polyfit_degree.
    """
    degree = polyfit_degree
    outer_deg = resolve_outer_degree(outer_degree)
    sf = fitter if fitter is not None else SpectrumFitter()

    # the template chain: centres from the measured profile itself
    # (template_calibration.py); the profiles ride along on the parameters
    # object for the sample kernels (not persisted).
    from brillouin_system.spectrum_fitting.template_calibration import (
        calibration_parameters_from_template)
    params, profiles = calibration_parameters_from_template(
        data, sf, int(sf.sline_config.n_peaks), degree,
        outer_degree=outer_deg)
    params.template_profiles = profiles
    return params
