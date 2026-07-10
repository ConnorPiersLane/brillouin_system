
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.spectrum_fitting.dho_model import ElasticAnchors
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

    # Empirical per-order instrument response (ePSF = optical PSF including
    # the 1-px binning), reconstructed from the reference sweep. Sampled on a
    # uniform grid of spacing psf_grid_step_px, symmetric around the profile
    # maximum at the center index, area-normalised. In a dual-chain
    # calibration these live on the psf_variant, not here.
    psf_left: Optional[np.ndarray] = field(default=None)
    psf_right: Optional[np.ndarray] = field(default=None)
    psf_grid_step_px: Optional[float] = field(default=None)

    # PSF-centered sibling chain, built by calibrate() whenever the sweep
    # allows the ePSF reconstruction: same reference sweep, but every center
    # re-fitted through the reconstructed ePSF (its psf_left / psf_right /
    # psf_grid_step_px are set, its own psf_variant is None). The top-level
    # fields above always hold the Lorentzian-centered chain. A fitted center
    # is only unbiased against the chain whose centering matches the sample
    # model, so CalibrationCalculator.for_chain selects the chain from the
    # fit's calibration_chain stamp. None when the reconstruction failed and
    # in files predating this field.
    psf_variant: Optional["CalibrationPolyfitParameters"] = field(default=None)


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
        self._variant_calc: CalibrationCalculator | None = None

    def _variant(self) -> "CalibrationCalculator | None":
        """Calculator over the PSF-centered chain, or None if not stored."""
        if self.p is None or self.p.psf_variant is None:
            return None
        if self._variant_calc is None:
            self._variant_calc = CalibrationCalculator(self.p.psf_variant)
        return self._variant_calc

    def for_chain(self, chain: str | None) -> "CalibrationCalculator":
        """
        Calculator over the calibration chain a fit was stamped with
        (FittedSpectrum.calibration_chain, set by SpectrumFitter.fit from
        ElasticAnchors.chain).

        "lorentzian" (or empty/None, covering fits saved before the stamp
        existed) selects the main chain; "psf" selects the PSF-centered
        variant. Mixing the chains reintroduces the sub-pixel bias the PSF
        calibration removes, so every px->GHz conversion of a fit result
        must go through this selector.

        A "psf" stamp against a calibration without a variant is a real
        mismatch — the fit consumed PSF anchors this calibration cannot have
        produced — so it raises instead of silently mapping the fit through
        the wrong polynomials.
        """
        if not chain or chain == "lorentzian":
            return self
        if chain == "psf":
            variant = self._variant()
            if variant is None:
                raise ValueError(
                    "Fit is stamped with the 'psf' calibration chain, but "
                    "this calibration has no PSF variant — the fit was made "
                    "against a different calibration."
                )
            return variant
        raise ValueError(f"Unknown calibration chain '{chain}'.")

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

        # Map the fitted pixels through the chain the fit was made against
        # (PSF-centered variant for *_psf / dho fits, main chain otherwise).
        calc = self.for_chain(fitting.calibration_chain)

        if reference == "left":
            px_value = fitting.left_peak_center_px
            if mode == "poly":
                result = calc.freq_left_peak(px_value)
            else:
                result = calc.freq_left_peak_interp(px_value)

        elif reference == "right":
            px_value = fitting.right_peak_center_px
            if mode == "poly":
                result = calc.freq_right_peak(px_value)
            else:
                result = calc.freq_right_peak_interp(px_value)

        elif reference == "distance":
            px_value = fitting.inter_peak_distance
            if mode == "poly":
                result = calc.freq_peak_distance(px_value)
            else:
                result = calc.freq_peak_distance_interp(px_value)

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

    @staticmethod
    def _elastic_line_px(coeffs, px_points, freq_points) -> float | None:
        """
        Pixel position of the elastic (Rayleigh) line for one peak, or None.

        Uses a local Newton step from the calibrated point closest to zero
        frequency: R = px* - nu(px*) / nu'(px*). This extrapolates with the
        locally correct slope over the shortest possible distance, so a mildly
        nonlinear dispersion introduces only a small bias. Falls back to
        polynomial roots when no calibration points are stored (only
        unambiguous for degree-1 fits).
        """
        if coeffs is None:
            return None
        coeffs = np.asarray(coeffs, dtype=float)
        if not np.all(np.isfinite(coeffs)):
            return None

        dcoeffs = np.polyder(coeffs, m=1)

        if px_points is not None and freq_points is not None and len(px_points) > 0:
            freqs = np.asarray(freq_points, dtype=float)
            pxs = np.asarray(px_points, dtype=float)
            idx = int(np.argmin(np.abs(freqs)))
            px_star = float(pxs[idx])

            slope = float(np.polyval(dcoeffs, px_star))
            if slope == 0.0 or not np.isfinite(slope):
                return None

            r = px_star - float(np.polyval(coeffs, px_star)) / slope
            if not np.isfinite(r):
                return None
            # The elastic line must lie outside the measured sideband range.
            if np.min(pxs) <= r <= np.max(pxs):
                return None
            return r

        # No stored points: only a degree-1 polynomial has an unambiguous root.
        if len(coeffs) == 2:
            a, b = float(coeffs[0]), float(coeffs[1])
            if a == 0.0:
                return None
            r = -b / a
            return r if np.isfinite(r) else None

        return None

    def _own_chain_anchors(self) -> ElasticAnchors | None:
        """
        Anchors of THIS calculator's own (top-level) chain: Rayleigh pixel
        positions extracted from its polynomials, plus its ePSFs when the
        chain carries them (the PSF variant; also legacy single-chain files
        that stored PSFs at the top level).
        """
        r_left = self._elastic_line_px(
            self.p.freq_left_peak, self.p.left_px_points, self.p.left_freq_points
        )
        r_right = self._elastic_line_px(
            self.p.freq_right_peak, self.p.right_px_points, self.p.right_freq_points
        )

        if r_left is None or r_right is None:
            return None
        if not r_left < r_right:
            return None

        psf_left = psf_right = None
        psf_step = None
        if (
            self.p.psf_left is not None
            and self.p.psf_right is not None
            and self.p.psf_grid_step_px
        ):
            pl = np.asarray(self.p.psf_left, dtype=float)
            pr = np.asarray(self.p.psf_right, dtype=float)
            if np.all(np.isfinite(pl)) and np.all(np.isfinite(pr)):
                psf_left, psf_right = pl, pr
                psf_step = float(self.p.psf_grid_step_px)

        return ElasticAnchors(
            rayleigh_left_px=r_left,
            rayleigh_right_px=r_right,
            psf_left=psf_left,
            psf_right=psf_right,
            psf_grid_step_px=psf_step,
            chain="lorentzian",
        )

    def elastic_anchors(self) -> ElasticAnchors | None:
        """
        Pixel positions of the elastic (Rayleigh) lines bracketing the two
        Brillouin peaks, or None if the calibration cannot provide them.

        Returns the MAIN-chain anchors (all the pure eq.-S2 dho needs), with
        the PSF-chain anchors nested as .psf_variant when the calibration
        stores the variant — mirroring the dual-chain parameters. The fitter
        selects the base anchors for the pure dho and the variant for the
        PSF-aware models (dho_psf, lorentzian_psf); each carries its chain of
        origin (ElasticAnchors.chain), which the fitter stamps onto the
        resulting FittedSpectrum so the analysis maps it through the same
        chain.

        Returns None only when no chain can provide anchors (anchor
        unavailable or geometry inconsistent). If the main chain cannot but
        the variant can, the variant anchors are returned directly.
        """
        if self.p is None:
            return None

        main = self._own_chain_anchors()

        variant_calc = self._variant()
        var_anchors = None
        if variant_calc is not None:
            var_anchors = variant_calc._own_chain_anchors()
            if var_anchors is not None:
                var_anchors.chain = "psf"

        if main is None:
            return var_anchors
        main.psf_variant = var_anchors
        return main

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
        anchors = self.elastic_anchors()
        if anchors is not None:
            lines.append(
                f"Elastic lines (DHO anchors): left ≈ {anchors.rayleigh_left_px:.2f} px, "
                f"right ≈ {anchors.rayleigh_right_px:.2f} px"
            )
        else:
            lines.append("Elastic lines (DHO anchors): not available")
        if self._variant() is not None:
            lines.append(
                "PSF-centered variant chain: available "
                "(used automatically by *_psf / dho sample models)"
            )
        else:
            lines.append(
                "PSF-centered variant chain: NOT available (ePSF "
                "reconstruction not usable) — *_psf sample models will "
                "refuse to fit against this calibration"
            )
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


def get_calibration_calculator_from_data(
    calibration_data: CalibrationData, poyfit_degree
) -> CalibrationCalculator:
    return CalibrationCalculator(
        calibrate(data=calibration_data, poyfit_degree=poyfit_degree)
    )

def sort_xy(x, y):
    idx = np.argsort(x)
    return np.asarray(x)[idx], np.asarray(y)[idx]

sf = SpectrumFitter()


# --- PSF (empirical instrument response) reconstruction -------------------
# The reference sidebands are quasi-monochromatic, so every reference
# spectrum is a 1-px-sampled copy of the per-order instrument response. The
# sweep slides it across the detector in sub-pixel steps, which allows a
# super-resolved reconstruction (astronomy "ePSF" technique). The
# reconstructed profile is the OPTICAL psf convolved with the 1-px binning
# (it is built from pixel-integrated samples), so models using it must
# sample at pixel centers WITHOUT integrating over the pixel again.

PSF_GRID_STEP_PX = 0.05
PSF_HALF_WIDTH_PX = 8.0


@dataclass
class _ReferenceFit:
    """One bootstrap-fitted reference spectrum."""
    px: np.ndarray
    sline: np.ndarray
    freq: float
    left_center_px: float
    right_center_px: float
    left_width_px: float
    right_width_px: float
    left_amplitude: float
    right_amplitude: float
    offset: float


def _bootstrap_reference_fits(data: CalibrationData) -> list[_ReferenceFit]:
    entries = []
    for freq_block in data.measured_freqs:
        for point in freq_block.cali_meas_points:
            px, sline = sf.get_px_sline_from_image(point.frame)
            fs = sf.fit(px, sline, is_reference_mode=True)
            if fs.is_success:
                entries.append(_ReferenceFit(
                    px=np.asarray(px, dtype=float),
                    sline=np.asarray(sline, dtype=float),
                    freq=float(point.microwave_freq),
                    left_center_px=float(fs.left_peak_center_px),
                    right_center_px=float(fs.right_peak_center_px),
                    left_width_px=float(fs.left_peak_width_px),
                    right_width_px=float(fs.right_peak_width_px),
                    left_amplitude=float(fs.left_peak_amplitude),
                    right_amplitude=float(fs.right_peak_amplitude),
                    offset=float(fs.offset),
                ))
    return entries


def _build_polyfit_parameters(entries: list[_ReferenceFit], degree: int) -> CalibrationPolyfitParameters:
    freqs_all = np.asarray([e.freq for e in entries], dtype=float)
    left_px = np.asarray([e.left_center_px for e in entries], dtype=float)
    right_px = np.asarray([e.right_center_px for e in entries], dtype=float)
    inter_px = right_px - left_px
    left_width = np.asarray([e.left_width_px for e in entries], dtype=float)
    right_width = np.asarray([e.right_width_px for e in entries], dtype=float)

    def safe_polyfit(x, y, deg):
        if len(x) <= deg:
            print(f"[Calibration Warning] Not enough points for degree {deg} fit (got {len(x)} points).")
            return np.full(deg + 1, np.nan)
        return np.polyfit(x, y, deg)

    by_freq: dict[float, list[int]] = {}
    for i, e in enumerate(entries):
        by_freq.setdefault(e.freq, []).append(i)

    freqs_mean, left_mean, right_mean, dist_mean = [], [], [], []
    for freq, idxs in by_freq.items():
        freqs_mean.append(freq)
        left_mean.append(float(np.mean(left_px[idxs])))
        right_mean.append(float(np.mean(right_px[idxs])))
        dist_mean.append(float(np.mean(inter_px[idxs])))

    left_px_sorted, left_freq_sorted = sort_xy(np.asarray(left_mean), np.asarray(freqs_mean))
    right_px_sorted, right_freq_sorted = sort_xy(np.asarray(right_mean), np.asarray(freqs_mean))
    dist_px_sorted, dist_freq_sorted = sort_xy(np.asarray(dist_mean), np.asarray(freqs_mean))

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


def _psf_grid(step: float = PSF_GRID_STEP_PX, half: float = PSF_HALF_WIDTH_PX) -> np.ndarray:
    n_bins = int(round(2.0 * half / step)) + 1
    return -half + step * np.arange(n_bins)


def _lorentzian_tail(px, amp, cen, hwhm):
    """Approximate profile of the OTHER peak, subtracted before PSF binning."""
    hwhm = max(float(hwhm), 1e-6)
    return amp * hwhm**2 / (np.square(px - cen) + hwhm**2)


def _reconstruct_epsf(entries: list[_ReferenceFit], coeffs, side: str) -> np.ndarray | None:
    """
    Reconstruct the ePSF of one order from the reference sweep.

    Alignment centers come from the (smooth) calibration polynomial evaluated
    at each frame's known microwave frequency — NOT from the per-frame fitted
    centers — so per-frame fit bias (the pixel-phase wiggle) does not distort
    the reconstruction. The other order's tail is subtracted using its
    bootstrap Lorentzian parameters.

    Returns an area-normalised profile on _psf_grid(), maximum at the center
    bin, or None if the reconstruction is not usable.
    """
    if coeffs is None:
        return None
    coeffs = np.asarray(coeffs, dtype=float)
    if not np.all(np.isfinite(coeffs)):
        return None
    dcoeffs = np.polyder(coeffs)

    step = PSF_GRID_STEP_PX
    half = PSF_HALF_WIDTH_PX
    grid = _psf_grid(step, half)
    n_bins = len(grid)
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    expected_px = int(2 * half) + 1

    for e in entries:
        if side == "left":
            c = e.left_center_px
            other_amp, other_cen, other_w = e.right_amplitude, e.right_center_px, e.right_width_px
        else:
            c = e.right_center_px
            other_amp, other_cen, other_w = e.left_amplitude, e.left_center_px, e.left_width_px

        # Newton-refine the alignment center on the smooth polynomial.
        for _ in range(3):
            s = float(np.polyval(dcoeffs, c))
            if s == 0.0 or not np.isfinite(s):
                c = np.nan
                break
            c = c - (float(np.polyval(coeffs, c)) - e.freq) / s
        if not np.isfinite(c):
            continue

        mask = np.abs(e.px - c) <= half
        if int(mask.sum()) < expected_px - 1:  # skip edge-truncated windows
            continue

        # The 2lorentzian amplitude is the raw kernel parameter; convert to
        # peak height for the tail estimate.
        other_w_safe = max(float(other_w), 1e-6)
        other_height = other_amp * other_w_safe * 2.0 * np.arctan(0.5 / other_w_safe)

        vals = e.sline[mask] - _lorentzian_tail(e.px[mask], other_height, other_cen, other_w)

        # Local baseline from the window edges: the bootstrap window-fit
        # offset is biased for near-pixel-wide peaks, and a wrong baseline
        # skews the reconstructed profile once wings are clipped.
        edge = np.abs(e.px[mask] - c) >= half - 2.0
        if int(edge.sum()) >= 3:
            baseline = float(np.median(vals[edge]))
        else:
            baseline = e.offset
        vals = vals - baseline

        area = float(vals.sum())
        if not np.isfinite(area) or area <= 0:
            continue
        vals = vals / area

        idx = np.round((e.px[mask] - c + half) / step).astype(int)
        ok = (idx >= 0) & (idx < n_bins)
        np.add.at(sums, idx[ok], vals[ok])
        np.add.at(counts, idx[ok], 1.0)

    filled = counts > 0
    if filled.sum() < 10:
        return None
    # Phase-coverage criterion: what matters is the effective sampling of the
    # profile, i.e. the largest gap between filled bins — a sweep whose pixel
    # step is (near-)rational fills few distinct sub-pixel phases but can
    # still sample densely enough. Reject only genuinely coarse coverage.
    fidx = np.flatnonzero(filled)
    max_gap_px = step * float(np.max(np.diff(fidx)))
    if max_gap_px > 0.3:
        return None

    prof = np.interp(grid, grid[filled], sums[filled] / counts[filled])
    prof = np.convolve(prof, np.ones(3) / 3.0, mode="same")  # light smoothing
    prof = np.clip(prof, 0.0, None)

    # Shift the maximum to u = 0 (parabolic sub-bin interpolation).
    i = int(np.argmax(prof))
    d = 0.0
    if 0 < i < n_bins - 1:
        denom = prof[i - 1] - 2.0 * prof[i] + prof[i + 1]
        if denom != 0.0:
            d = float(np.clip(0.5 * (prof[i - 1] - prof[i + 1]) / denom, -1.0, 1.0))
    peak_u = grid[i] + d * step
    prof = np.interp(grid + peak_u, grid, prof, left=0.0, right=0.0)

    total = prof.sum()
    if total <= 0:
        return None
    return prof / total


def _fit_center_with_epsf(px, sline, psf, center0, amp0, offset0) -> float | None:
    """Fit amp * ePSF(px - center) + offset around center0; return center."""
    grid = _psf_grid()
    half = PSF_HALF_WIDTH_PX

    mask = np.abs(px - center0) <= half
    if int(mask.sum()) < 7:
        return None
    x = px[mask]
    y = sline[mask]

    def model(xx, amp, cen, off):
        return amp * np.interp(xx - cen, grid, psf, left=0.0, right=0.0) + off

    psf_max = float(np.max(psf))
    p0 = [max(amp0, 1e-6) / max(psf_max, 1e-12), center0, offset0]
    try:
        popt, _ = curve_fit(
            model, x, y, p0=p0,
            bounds=([0.0, center0 - half, -np.inf], [np.inf, center0 + half, np.inf]),
            maxfev=20000,
        )
    except Exception:
        return None
    return float(popt[1])


def _refit_entries_with_epsf(entries, psf_left, psf_right) -> list[_ReferenceFit]:
    refit = []
    for e in entries:
        c_left = _fit_center_with_epsf(e.px, e.sline, psf_left, e.left_center_px, e.left_amplitude, e.offset)
        c_right = _fit_center_with_epsf(e.px, e.sline, psf_right, e.right_center_px, e.right_amplitude, e.offset)
        refit.append(_ReferenceFit(
            px=e.px,
            sline=e.sline,
            freq=e.freq,
            left_center_px=c_left if c_left is not None else e.left_center_px,
            right_center_px=c_right if c_right is not None else e.right_center_px,
            left_width_px=e.left_width_px,
            right_width_px=e.right_width_px,
            left_amplitude=e.left_amplitude,
            right_amplitude=e.right_amplitude,
            offset=e.offset,
        ))
    return refit


def calibrate(
    data: CalibrationData,
    poyfit_degree: int = 1,
) -> CalibrationPolyfitParameters:
    """
    Build dual-chain calibration parameters from reference sweep data.

    The returned top-level parameters are ALWAYS the Lorentzian-centered
    chain (reference peak centers from lorentzian_window fits — the classic
    pipeline, paired with the plain sample models).

    Whenever the sweep allows it, the PSF-centered sibling chain is built and
    attached as .psf_variant: reconstruct the per-order empirical PSF from
    the sweep, re-fit every reference center with a shifted-PSF model
    (immune to PSF skew and sub-pixel sampling bias), rebuild the
    polynomials from the improved centers, and store the PSFs on the variant
    for the PSF-aware sample fits (lorentzian_psf, dho).
    CalibrationCalculator then selects the chain per fit, so either model
    family can be analyzed from the same calibration. Falls back to the
    main chain alone (with a printed warning) if the reconstruction is
    unusable — the *_psf sample models then refuse to fit.
    """
    entries = _bootstrap_reference_fits(data)
    if not entries:
        raise ValueError("No successful fits found in calibration data.")

    params = _build_polyfit_parameters(entries, poyfit_degree)

    # Pass 1: reconstruct the ePSFs aligned with the bootstrap polynomials.
    psf_left = _reconstruct_epsf(entries, params.freq_left_peak, side="left")
    psf_right = _reconstruct_epsf(entries, params.freq_right_peak, side="right")
    if psf_left is None or psf_right is None:
        print(
            "[Calibration] PSF reconstruction not usable — main (lorentzian) "
            "chain only; *_psf / dho-with-PSF sample models will not be "
            "available for this calibration."
        )
        return params

    # Re-fit all reference centers with the shifted-PSF model and rebuild.
    refit_entries = _refit_entries_with_epsf(entries, psf_left, psf_right)
    params_psf = _build_polyfit_parameters(refit_entries, poyfit_degree)

    # Pass 2: re-reconstruct with the improved polynomials (better alignment).
    psf_left2 = _reconstruct_epsf(entries, params_psf.freq_left_peak, side="left")
    psf_right2 = _reconstruct_epsf(entries, params_psf.freq_right_peak, side="right")

    params_psf.psf_left = psf_left2 if psf_left2 is not None else psf_left
    params_psf.psf_right = psf_right2 if psf_right2 is not None else psf_right
    params_psf.psf_grid_step_px = PSF_GRID_STEP_PX

    # Dual-chain result: the Lorentzian bootstrap chain stays the main
    # parameters (old semantics, pairs with the plain sample models); the
    # PSF-centered chain rides along as the variant.
    params.psf_variant = params_psf
    return params
