import numpy as np
from scipy.optimize import curve_fit

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    FindPeaksConfig,
    find_peaks_sample_config,
    find_peaks_reference_config,
    SlineFromFrameConfig,
    sline_from_frame_config,
    FittingConfigs,
    MODEL_PRESETS,
    NA_WEIGHTINGS,
)
from brillouin_system.spectrum_fitting.fit_util import (
    find_peak_locations,
    select_top_two_peaks,
    sort_peaks,
    refine_fitted_spectrum,
)

from brillouin_system.spectrum_fitting.voigt_model import (
    _1voigt_binned,
    _2voigt_binned,
    _voigt_pixel_integrated,
)
from brillouin_system.spectrum_fitting.pixel_response import pixel_response_profile
from brillouin_system.spectrum_fitting.row_selection import (
    select_rows,
    captured_fraction,
)

from brillouin_system.spectrum_fitting.elastic_anchors import ElasticAnchors
from brillouin_system.spectrum_fitting.na_correction5 import gaussian_angle_width
from brillouin_system.spectrum_fitting.na_lineshape import make_2na_lorentzian_binned

# A model name selects the LINESHAPE only. Windowing (config.use_window) and
# the baseline (config.background) are independent options that apply to any
# lineshape.
SUPPORTED_MODELS = (
    "lorentzian",
    "voigt",
    "pixel_response",
    "na_lorentzian",
    "na_gauss_lorentzian",
)

# Models that fit the NA-integrated lineshape: each Brillouin peak is anchored
# at its own Rayleigh-order elastic line, so they need ElasticAnchors from the
# calibration (see model_requires_anchors below).
# The collection weight over the cone is the config toggle na_weighting
# ("uniform" = hard pupil only, the NA 0.14 recipe; "uniform_gaussian" adds
# the Gaussian fiber-coupling apodization from na_beam_diameter_mm /
# na_focal_length_mm, the NA 0.42 recipe). The legacy 'na_gauss_lorentzian'
# name is still accepted for callers that assign config.fitting_model directly
# and forces the Gaussian weighting.
NA_GAUSS_MODELS = ("na_gauss_lorentzian",)
NA_MODELS = ("na_lorentzian",) + NA_GAUSS_MODELS


def normalize_model_name(model: str):
    """Accept the retired '<model>_window' names and the prm* presets.

    FindPeaksConfig normalises these on construction, but callers that assign
    config.fitting_model directly bypass __post_init__, so the fitter tolerates
    them too. Returns (base_name, window_forced).
    """
    model = str(model)
    if model.endswith("_window"):
        return model[: -len("_window")], True
    if model in MODEL_PRESETS:
        return MODEL_PRESETS[model]["fitting_model"], True
    return model, False


def model_requires_anchors(model: str) -> bool:
    """True if the fitting model needs ElasticAnchors from a calibration."""
    return normalize_model_name(model)[0] in NA_MODELS


def is_pixel_response_fit(model: str | None) -> bool:
    """True if a FittedSpectrum came from the pixel-response lineshape.

    Takes the `model` string a fit carries (the fit_kind tag, e.g.
    '2pixel_response_window_linear_per_peak'), not a config model name.
    """
    return "pixel_response" in str(model or "")


# -----------------------------
# Symmetric Lorentzian models
# -----------------------------

def _lorentzian_pixel_integrated(x, amp, cen, wid):
    x = np.asarray(x, dtype=float)
    wid = max(float(wid), 1e-12)
    left = x - 0.5
    right = x + 0.5
    return amp * wid * (
        np.arctan((right - cen) / wid)
        - np.arctan((left - cen) / wid)
    )


def _1lorentzian_binned(x, amp, cen, wid, offset):
    return _lorentzian_pixel_integrated(x, amp, cen, wid) + offset


def _2lorentzian_binned(x, amp1, cen1, wid1, amp2, cen2, wid2, offset):
    return (
        _lorentzian_pixel_integrated(x, amp1, cen1, wid1)
        + _lorentzian_pixel_integrated(x, amp2, cen2, wid2)
        + offset
    )


def _asym_lorentzian_pixel_integrated(x, amp, cen, wid_left, wid_right):
    """Pixel-integrated two-half-width Lorentzian.

    NOT a selectable model any more (tested on calibration data 2026-07 and
    rejected — it triples the residual sine). Kept as a plain function because
    the analysis scripts in Dropbox Data/2026-7-28/fixed_skew/ use it to build
    synthetic skewed truths.
    """
    x = np.asarray(x, dtype=float)

    left = x - 0.5
    right = x + 0.5

    wid_left = max(float(wid_left), 1e-12)
    wid_right = max(float(wid_right), 1e-12)

    y = np.zeros_like(x, dtype=float)

    m_left = right <= cen
    y[m_left] = amp * wid_left * (
        np.arctan((right[m_left] - cen) / wid_left)
        - np.arctan((left[m_left] - cen) / wid_left)
    )

    m_right = left >= cen
    y[m_right] = amp * wid_right * (
        np.arctan((right[m_right] - cen) / wid_right)
        - np.arctan((left[m_right] - cen) / wid_right)
    )

    m_cross = ~(m_left | m_right)
    y[m_cross] = (
        amp * wid_left * np.arctan((cen - left[m_cross]) / wid_left)
        + amp * wid_right * np.arctan((right[m_cross] - cen) / wid_right)
    )

    return y


# -----------------------------
# Baselines
# -----------------------------
# The background is independent of the lineshape. A model function is built
# from a peak part followed by the baseline parameters, always appended LAST so
# the peak parameters keep a fixed layout.


def _background_masks(x, centers):
    """Split a pixel axis between peaks for the per-peak baseline.

    Each peak owns the points nearest to it (split at the midpoint), so its
    constant+slope describe the LOCAL gradient under that peak rather than the
    level difference between the two windows. Membership is computed from the
    coordinates on every call, so the same background function works on the
    fit window, on the full pixel axis and on the refined grid.
    """
    x = np.asarray(x, dtype=float)
    centers = [float(c) for c in centers]
    if len(centers) == 1:
        return [np.ones_like(x, dtype=bool)]
    mid = 0.5 * (centers[0] + centers[1])
    left = x <= mid
    return [left, ~left]


def _make_background(background: str, px_fit, centers, offset0):
    """Return (func(x, *bg_params), p0, lo, hi, n_params).

    The per-peak masks are bound at build time, so the model must be rebuilt if
    the fit domain changes.
    """
    if background == "flat":
        # Lower bound 0 preserves the long-standing behaviour (the sline is
        # clipped to >= 0, so a negative pedestal is not physical here).
        def func(x, off):
            return off
        return func, [offset0], [0.0], [np.inf], 1

    x0 = float(np.mean(px_fit))

    if background == "linear":
        def func(x, off, slope):
            return off + slope * (np.asarray(x, dtype=float) - x0)
        return func, [offset0, 0.0], [-np.inf, -np.inf], [np.inf, np.inf], 2

    if background == "flat_per_peak":
        # Per-peak constant offset, no slope: the width-safe per-peak baseline
        # (a freed slope is odd-symmetric and leaks into the fitted width via
        # covariance — see BACKGROUNDS in find_peaks_config).
        n_parts = len(_background_masks(px_fit, centers))

        def func(x, *params):
            x = np.asarray(x, dtype=float)
            out = np.zeros_like(x)
            for i, m in enumerate(_background_masks(x, centers)):
                out = out + m * params[i]
            return out

        return (func, [offset0] * n_parts, [0.0] * n_parts,
                [np.inf] * n_parts, n_parts)

    if background == "linear_per_peak":
        # Reference points are fixed from the fit domain so the parameters keep
        # a stable meaning, but membership is recomputed from x on each call.
        fit_masks = _background_masks(px_fit, centers)
        x0s = [float(np.mean(px_fit[m])) if np.any(m) else x0 for m in fit_masks]
        n_parts = len(fit_masks)

        def func(x, *params):
            x = np.asarray(x, dtype=float)
            out = np.zeros_like(x)
            for i, m in enumerate(_background_masks(x, centers)):
                out = out + m * (params[2 * i] + params[2 * i + 1] * (x - x0s[i]))
            return out

        n = 2 * n_parts
        p0 = []
        for _ in range(n_parts):
            p0 += [offset0, 0.0]
        return func, p0, [-np.inf] * n, [np.inf] * n, n

    raise ValueError(f"Unknown background '{background}'.")


class SpectrumFitter:
    def __init__(self):
        self.sline_config: SlineFromFrameConfig = sline_from_frame_config.get()
        self.sample_config: FindPeaksConfig = find_peaks_sample_config.get()
        self.reference_config: FindPeaksConfig = find_peaks_reference_config.get()
        # Rows chosen by the automatic band selection, frozen after the first
        # use so the band cannot drift between a scan's calibration and its
        # samples (a one-row difference biases the two peaks by ~3-4 MHz in
        # opposite directions).
        self._auto_rows: list[int] | None = None

    def update_configs(self, configs: FittingConfigs):
        self.update_sline_config(configs.sline_config)
        self.update_sample_config(configs.sample_config)
        self.update_reference_config(configs.reference_config)

    def update_sline_config(self, sline_config: SlineFromFrameConfig):
        if not isinstance(sline_config, SlineFromFrameConfig):
            raise TypeError("sline_config must be a SlineFromFrame instance.")
        self.sline_config = sline_config
        # A new sline config may change n_rows / the mode: re-locate the band.
        self._auto_rows = None

    def auto_select_rows(self, frames) -> list[int]:
        """Locate and freeze the row band from a frame or a stack of frames.

        Call this ONCE per scan with a representative stack (more frames = a
        better-determined centroid), then use the same fitter — or the same
        rows — for that scan's calibration and sample frames. The chosen rows
        are returned so they can be stored with the data.
        """
        rows = select_rows(frames, self.sline_config.n_rows)
        self._auto_rows = rows
        print(f"[SpectrumFitter] auto row band: {rows[0]}-{rows[-1]} "
              f"({len(rows)} rows, "
              f"{100 * captured_fraction(frames, rows):.1f}% of the signal)")
        return rows

    def get_selected_rows(self, frame: np.ndarray | None = None) -> list[int]:
        """Rows summed into the spectral line, honouring the selection mode."""
        if self.sline_config.row_selection != "auto":
            return list(self.sline_config.selected_rows)
        if self._auto_rows is None:
            if frame is None:
                raise ValueError(
                    "row_selection is 'auto' but no band has been located yet; "
                    "call auto_select_rows(frames) first."
                )
            # Freeze on first use so the band stays put for the rest of the
            # scan. Prefer calling auto_select_rows() with a stack.
            self.auto_select_rows(frame)
        return list(self._auto_rows)

    def update_sample_config(self, sample_config: FindPeaksConfig):
        if not isinstance(sample_config, FindPeaksConfig):
            raise TypeError("sample_config must be a FindPeaksConfig instance.")
        self.sample_config = sample_config

    def update_reference_config(self, reference_config: FindPeaksConfig):
        if not isinstance(reference_config, FindPeaksConfig):
            raise TypeError("reference_config must be a FindPeaksConfig instance.")
        self.reference_config = reference_config

    def get_px_sline_from_image(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_offset = self.sline_config.pixel_offset_left
        right_offset = self.sline_config.pixel_offset_right

        rows = self.get_selected_rows(frame)
        height = frame.shape[0]

        if not rows or not all(0 <= r < height for r in rows):
            print("[get_sline_from_image] Warning: Invalid or empty row list — using full image height.")
            rows = list(range(height))

        sline = frame[rows, :].sum(axis=0)
        px = np.arange(sline.shape[0])

        if right_offset > 0:
            sline = sline[left_offset:-right_offset]
            px = px[left_offset:-right_offset]
        else:
            sline = sline[left_offset:]
            px = px[left_offset:]

        return px, sline

    def get_empty_fitting(self, px, sline) -> FittedSpectrum:
        return FittedSpectrum(
            is_success=False,
            x_pixels=px,
            sline=sline,
        )

    def get_total_sline_value(self, sline) -> float:
        if sline is None:
            return 0.0
        return float(np.sum(sline))

    # -----------------------------
    # Lineshape assembly
    # -----------------------------

    def _peak_model(
        self, model, config, n_peaks, amp, cen, wid,
        center_ranges, x_span, use_window, anchors, alpha, na_v0,
    ):
        """Build the peak part of the model.

        Returns (func(x, *peak_params), p0, lo, hi, n_params_per_peak).
        Peaks are always ordered left-to-right, so per-peak options (the
        pixel-response tails, the per-peak baseline) line up with them.
        """
        def width_bounds(i, floor=1e-12, voigt=False):
            if not use_window:
                return (0.03 if voigt else floor), x_span / 2
            lo_w = max(0.03 if voigt else 1e-6, 0.25 * float(wid[i]))
            return lo_w, max(lo_w * 2, 4.0 * float(wid[i]))

        if model in ("lorentzian", "pixel_response"):
            if model == "pixel_response":
                sigma = float(config.pr_sigma_px)
                tau_l = float(config.pr_tau_left_px)
                tau_r = float(config.pr_tau_right_px)
                if sigma <= 0.0 and tau_l <= 0.0 and tau_r <= 0.0:
                    raise ValueError(
                        "Model 'pixel_response' requires the frozen camera "
                        "constants: set pr_sigma_px and/or pr_tau_left_px / "
                        "pr_tau_right_px in the find-peaks config. Measured "
                        "2026-07: 0.25 / 0.40 / 0.20 px. With all three at 0 "
                        "this model is just 'lorentzian'."
                    )
                taus = [tau_l, tau_r]

                def peak(x, a, c, w, i):
                    return pixel_response_profile(x, a, c, w, sigma, taus[i])
            else:
                def peak(x, a, c, w, i):
                    return _lorentzian_pixel_integrated(x, a, c, w)

            if n_peaks == 1:
                def func(x, a, c, w):
                    return peak(x, a, c, w, 0)
                lo_w, hi_w = width_bounds(0)
                return (func, [amp[0], cen[0], wid[0]],
                        [0, center_ranges[0][0], lo_w],
                        [np.inf, center_ranges[0][1], hi_w], 3)

            def func(x, a1, c1, w1, a2, c2, w2):
                return peak(x, a1, c1, w1, 0) + peak(x, a2, c2, w2, 1)

            lo0, hi0 = width_bounds(0)
            lo1, hi1 = width_bounds(1)
            return (
                func,
                [amp[0], cen[0], wid[0], amp[1], cen[1], wid[1]],
                [0, center_ranges[0][0], lo0, 0, center_ranges[1][0], lo1],
                [np.inf, center_ranges[0][1], hi0,
                 np.inf, center_ranges[1][1], hi1],
                3,
            )

        if model == "voigt":
            if n_peaks == 1:
                lo_w, hi_w = width_bounds(0, voigt=True)

                def func(x, a, c, g, s):
                    return _voigt_pixel_integrated(x, a, c, g, s)

                return (func, [amp[0], cen[0], wid[0], 0.25],
                        [0, center_ranges[0][0], lo_w, 0.0],
                        [np.inf, center_ranges[0][1], hi_w, 5.0], 4)

            def func(x, a1, c1, g1, s1, a2, c2, g2, s2):
                return (_voigt_pixel_integrated(x, a1, c1, g1, s1)
                        + _voigt_pixel_integrated(x, a2, c2, g2, s2))

            lo0, hi0 = width_bounds(0, voigt=True)
            lo1, hi1 = width_bounds(1, voigt=True)
            return (
                func,
                [amp[0], cen[0], wid[0], 0.25, amp[1], cen[1], wid[1], 0.25],
                [0, center_ranges[0][0], lo0, 0.0,
                 0, center_ranges[1][0], lo1, 0.0],
                [np.inf, center_ranges[0][1], hi0, 5.0,
                 np.inf, center_ranges[1][1], hi1, 5.0],
                4,
            )

        if model in NA_MODELS:
            na_func = make_2na_lorentzian_binned(
                anchors.rayleigh_left_px, anchors.rayleigh_right_px,
                alpha, v0=na_v0,
            )

            def func(x, a1, c1, w1, a2, c2, w2):
                return na_func(x, a1, c1, w1, a2, c2, w2, 0.0)

            lo0, hi0 = width_bounds(0)
            lo1, hi1 = width_bounds(1)
            return (
                func,
                [amp[0], cen[0], wid[0], amp[1], cen[1], wid[1]],
                [0, center_ranges[0][0], lo0, 0, center_ranges[1][0], lo1],
                [np.inf, center_ranges[0][1], hi0,
                 np.inf, center_ranges[1][1], hi1],
                3,
            )

        raise ValueError(f"Unknown model: '{model}'.")

    def fit(
        self,
        px: np.ndarray,
        sline: np.ndarray,
        is_reference_mode: bool,
        anchors: ElasticAnchors | None = None,
    ) -> FittedSpectrum:
        config = self.reference_config if is_reference_mode else self.sample_config
        requested_model, window_forced = normalize_model_name(config.fitting_model)
        use_window = bool(getattr(config, "use_window", True)) or window_forced
        background = getattr(config, "background", "flat")
        beta = config.beta

        # Presets pin background and beta too (callers that assign
        # config.fitting_model directly bypass FindPeaksConfig.__post_init__).
        preset = MODEL_PRESETS.get(str(config.fitting_model))
        if preset is not None:
            background = preset["background"]
            beta = preset["beta"]

        if requested_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model: '{requested_model}'. "
                f"Supported models are {', '.join(SUPPORTED_MODELS)}. "
                f"Windowing and the baseline are separate config options "
                f"(use_window, background), not part of the model name."
            )

        # The model-mixing trap: 'pixel_response' defines the peak centre as
        # the Lorentzian core BEFORE the asymmetric tail, ~0.27 px away from a
        # plain Lorentzian's apparent centre. Fitting samples with one
        # convention against a calibration fitted with the other injects a
        # -168 MHz left-right split (measured 2026-08). Calibration and
        # samples must therefore use the same lineshape family.
        if not is_reference_mode:
            reference_model, _ = normalize_model_name(
                self.reference_config.fitting_model)
            pr = "pixel_response"
            if (pr in (requested_model, reference_model)
                    and requested_model != reference_model):
                raise ValueError(
                    f"Model mixing: sample model '{requested_model}' with "
                    f"reference model '{reference_model}'. The pixel-response "
                    f"centre convention differs from a plain Lorentzian's by "
                    f"~0.27 px (-168 MHz split when mixed), so calibration and "
                    f"samples must both use 'pixel_response' (e.g. prm0/prm1) "
                    f"or neither."
                )
            if requested_model == pr == reference_model:
                ours = (config.pr_sigma_px, config.pr_tau_left_px,
                        config.pr_tau_right_px)
                theirs = (self.reference_config.pr_sigma_px,
                          self.reference_config.pr_tau_left_px,
                          self.reference_config.pr_tau_right_px)
                if ours != theirs:
                    raise ValueError(
                        f"Camera-constant mismatch: sample config has "
                        f"(sigma, tau_l, tau_r) = {ours} but reference config "
                        f"has {theirs}. Different kernels define different "
                        f"peak centres, so calibration and samples must use "
                        f"identical pr_* constants — edit both sections of "
                        f"the find-peaks config."
                    )

        alpha = None
        na_v0 = None
        if requested_model in NA_MODELS:
            if anchors is None:
                raise ValueError(
                    f"Model '{requested_model}' requires elastic anchors from a "
                    f"calibration (CalibrationCalculator.elastic_anchors()); none were "
                    f"provided — is a calibration loaded?"
                )
            # The collection weight comes from the config toggle; the legacy
            # 'na_gauss_lorentzian' model name forces the Gaussian weighting.
            if requested_model in NA_GAUSS_MODELS:
                na_weighting = "uniform_gaussian"
            else:
                na_weighting = str(getattr(config, "na_weighting", "uniform"))
            if na_weighting not in NA_WEIGHTINGS:
                raise ValueError(
                    f"Unknown na_weighting '{na_weighting}'. "
                    f"Choose one of {NA_WEIGHTINGS}."
                )
            na = float(config.na_collection)
            n_sample = float(config.na_n_sample)
            if not 0.0 < na < n_sample:
                raise ValueError(
                    f"Model '{requested_model}' requires the collection NA (aperture "
                    f"clip): set na_collection (0 < NA < n_sample) in the find-peaks "
                    f"sample config (got na_collection={na}, na_n_sample={n_sample})."
                )
            alpha = float(np.arcsin(na / n_sample))
            if na_weighting == "uniform_gaussian":
                beam_d = float(config.na_beam_diameter_mm)
                focal = float(config.na_focal_length_mm)
                if beam_d <= 0.0 or focal <= 0.0:
                    raise ValueError(
                        f"na_weighting 'uniform_gaussian' requires the Gaussian "
                        f"coupling geometry: set na_beam_diameter_mm (collection-fiber "
                        f"mode at the pupil) and na_focal_length_mm (objective) in the "
                        f"find-peaks sample config (got beam={beam_d}, focal={focal})."
                    )
                na_v0 = float(gaussian_angle_width(beam_d, focal, n_sample))

        px = np.asarray(px, dtype=np.float64)
        sline = np.asarray(sline, dtype=np.float64)

        finite_mask = np.isfinite(px) & np.isfinite(sline)
        px = px[finite_mask]
        sline = sline[finite_mask]

        # Keep this if your peak finder expects non-negative data. Remove if you want
        # the offset/background model to handle negative baseline excursions.
        sline = np.clip(sline, 0, None)

        pk_ind, pk_info = find_peak_locations(sline, config=config)
        if len(pk_ind) < 1:
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=requested_model,
            )

        pk_ind, pk_info = select_top_two_peaks(pk_ind, pk_info)
        amp, cen, wid = self._extract_peak_params(pk_ind, pk_info, px, sline)

        n_peaks = len(cen)
        if n_peaks < 1:
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=requested_model,
            )

        if requested_model in NA_MODELS and n_peaks < 2:
            # The NA model pairs each peak with its own Rayleigh order; with a
            # single detected peak the pairing is ambiguous.
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=requested_model,
            )

        # Peaks are ordered left-to-right from here on: the NA anchors, the
        # per-peak pixel-response tails and the per-peak baseline all rely on
        # it (select_top_two_peaks orders by height).
        if n_peaks == 2 and cen[1] < cen[0]:
            amp, cen, wid = amp[::-1], cen[::-1], wid[::-1]

        x_min = float(np.min(px))
        x_max = float(np.max(px))
        x_span = max(x_max - x_min, 1.0)
        offset0 = float(np.amin(sline))

        if use_window:
            mask = self._build_window_mask(px, cen, wid, beta=beta)
            center_ranges = self._bounded_center_ranges(px, cen, wid, beta=beta)
        else:
            mask = np.ones_like(px, dtype=bool)
            center_ranges = [(x_min, x_max)] * n_peaks

        px_fit = px[mask]
        sline_fit = sline[mask]

        peak_func, p0_pk, lo_pk, hi_pk, n_per_peak = self._peak_model(
            requested_model, config, n_peaks, amp, cen, wid,
            center_ranges, x_span, use_window, anchors, alpha, na_v0,
        )
        bg_func, p0_bg, lo_bg, hi_bg, n_bg = _make_background(
            background, px_fit, cen, offset0,
        )

        n_pk = len(p0_pk)

        def model_func(x, *params):
            return peak_func(x, *params[:n_pk]) + bg_func(x, *params[n_pk:])

        p0 = list(p0_pk) + list(p0_bg)
        bounds = (list(lo_pk) + list(lo_bg), list(hi_pk) + list(hi_bg))

        # Report which NA kernel was actually fitted, so the choice survives in
        # the saved data even when it came from the na_weighting toggle.
        kind_model = requested_model
        if requested_model in NA_MODELS:
            kind_model = ("na_gauss_lorentzian" if na_v0 is not None
                          else "na_lorentzian")
        fit_kind = f"{n_peaks}{kind_model}"
        if use_window:
            fit_kind += "_window"
        if background != "flat":
            fit_kind += f"_{background}"

        try:
            popt, _ = curve_fit(
                model_func,
                px_fit,
                sline_fit,
                p0=p0,
                bounds=bounds,
                method="trf",
                maxfev=50000,
            )
        except Exception as e:
            print(f"[SpectrumFitter] Fit failed: {e}")
            return FittedSpectrum(
                is_success=False,
                sline=sline,
                x_pixels=px,
                model=fit_kind,
            )

        peak_params = [list(popt[i * n_per_peak:(i + 1) * n_per_peak])
                       for i in range(n_peaks)]
        bg_params = list(popt[n_pk:])

        if n_peaks == 2 and peak_params[1][1] < peak_params[0][1]:
            if requested_model in NA_MODELS:
                # Peaks are tied to their Rayleigh anchors, so they cannot be
                # reordered; crossed centres mean the fit wandered off.
                print("[SpectrumFitter] NA fit failed: peaks crossed their anchor ordering.")
                return FittedSpectrum(
                    is_success=False,
                    sline=sline,
                    x_pixels=px,
                    model=fit_kind,
                )
            peak_params = peak_params[::-1]
            if background in ("flat_per_peak", "linear_per_peak"):
                k = n_bg // 2
                bg_params = bg_params[k:] + bg_params[:k]

        centers = [p[1] for p in peak_params]
        bg_at_peaks = bg_func(np.asarray(centers, dtype=float), *bg_params)
        offset_value = float(np.mean(np.atleast_1d(bg_at_peaks)))

        return self._build_result(
            px=px,
            sline=sline,
            model_func=model_func,
            popt=popt,
            model=fit_kind,
            mask=mask,
            peak_params=peak_params,
            offset_value=offset_value,
        )

    def _extract_peak_params(self, pk_ind, pk_info, px, sline):
        pk_ind = np.asarray(pk_ind, dtype=int)

        if len(pk_ind) < 1:
            return np.array([]), np.array([]), np.array([])

        widths_idx = 0.5 * np.asarray(pk_info["widths"], dtype=float)
        heights = np.asarray(pk_info["peak_heights"], dtype=float)

        if "left_ips" in pk_info and "right_ips" in pk_info:
            idx_axis = np.arange(len(px), dtype=float)
            centers_idx = 0.5 * (
                np.asarray(pk_info["left_ips"], dtype=float)
                + np.asarray(pk_info["right_ips"], dtype=float)
            )
            centers = np.interp(centers_idx, idx_axis, px)
        else:
            centers = px[np.clip(pk_ind, 0, len(px) - 1)].astype(float)

        widths = widths_idx * 1.0
        return heights, centers, widths

    def _build_result(self, px, sline, model_func, popt, model: str,
                      mask: np.ndarray, peak_params, offset_value: float) -> FittedSpectrum:
        """Assemble the result from peaks already parsed left-to-right.

        peak_params[i] is that peak's parameter list; the first three entries
        are always (amplitude, centre, width) for every lineshape.
        """
        fitted = model_func(px, *popt)
        x_fit, y_fit = refine_fitted_spectrum(model_func, px, popt, factor=10)

        left = peak_params[0]
        right = peak_params[-1]
        two_peaks = len(peak_params) == 2

        return FittedSpectrum(
            is_success=True,
            model=model,
            sline=sline,
            x_pixels=px,
            fitted_spectrum=fitted,
            x_fit_refined=x_fit,
            y_fit_refined=y_fit,
            mask_for_fitting=mask,
            parameters=popt,
            left_peak_center_px=float(left[1]),
            left_peak_width_px=float(left[2]),
            left_peak_amplitude=float(left[0]) if two_peaks else float(left[0] / 2.0),
            right_peak_center_px=float(right[1]),
            right_peak_width_px=float(right[2]),
            right_peak_amplitude=float(right[0]) if two_peaks else float(right[0] / 2.0),
            inter_peak_distance=abs(float(right[1]) - float(left[1])) if two_peaks else 0.0,
            offset=float(offset_value),
        )

    @staticmethod
    def _build_window_mask(px, centers, widths, beta=4.0):
        mask = np.zeros_like(px, dtype=bool)
        for c, w in zip(centers, widths):
            center_idx = int(np.argmin(np.abs(px - float(c))))
            half = max(int(round(beta * float(w))), 1)
            lo = max(center_idx - half, 0)
            hi = min(center_idx + half + 1, len(px))
            mask[lo:hi] = True
        return mask

    @staticmethod
    def _bounded_center_ranges(px, centers, widths, beta=4.0):
        x_min, x_max = float(np.min(px)), float(np.max(px))
        ranges = []
        for c, w in zip(centers, widths):
            c = float(c)
            half = beta * float(w)
            lo = max(x_min, c - half)
            hi = min(x_max, c + half)
            ranges.append((lo, hi))
        return ranges
