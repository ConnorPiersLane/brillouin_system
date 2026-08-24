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
    resolve_fit_options,
)
from brillouin_system.spectrum_fitting.fit_util import (
    find_peak_locations,
    select_top_n_peaks,
    refine_fitted_spectrum,
)

from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.spectrum_fitting.psf import psf_profile
from brillouin_system.spectrum_fitting.row_selection import (
    select_rows,
    captured_fraction,
)

log = get_logger(__name__)

# A model name selects the LINESHAPE only. Windowing (config.use_window) and
# the baseline (config.background) are independent options that apply to any
# lineshape. The NA-integrated lineshape models were removed 2026-08-20: the
# production high-NA recipe is the POST-HOC scalar correction (fit as at low
# NA, then divide by na_lineshape.na_mean_shift_ratio) — never in the fit.
SUPPORTED_MODELS = (
    "lorentzian",
    "lorentzian_x_psf",
)


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


def resolved_background(config) -> str:
    """The background a fit will actually use for this config.

    Preset-aware: callers that assign config.fitting_model directly bypass
    FindPeaksConfig.__post_init__, so a preset name can sit there with a stale
    config.background — resolve_fit_options handles that, and fit() resolves
    through the same path.
    """
    return resolve_fit_options(config).background


def config_requires_reflection_background(config) -> bool:
    """True if fits with this config need the mapped reflection background.

    Callers then build it per scan: ReflectionBackgroundMapper(
    get_current_background(), calibration, n_rows).render(px) and pass the
    result to fit() as reflection_background — when get_current_background()
    is None (nothing loaded, no fallback), pass None and fit() warns and
    degrades to per-peak flat offsets.
    """
    return resolved_background(config) == "reflection"


# One-shot: a prmr fit without a loaded template degrades to per-peak flat
# offsets. Warn once per process, not once per frame — a scan would repeat
# it hundreds of times.
_missing_reflection_bg_warned = False


def _warn_missing_reflection_background():
    global _missing_reflection_bg_warned
    if _missing_reflection_bg_warned:
        return
    _missing_reflection_bg_warned = True
    log.warning(
        "[SpectrumFitter] Background 'reflection' (prmr) requested but NO "
        "reflection background is loaded — fitting with per-peak flat "
        "offsets only (there is deliberately no default template: alignments "
        "differ). Load one in the analyzer ('Load Background') or record a "
        "'reflection_background' scan at the reflection plane and build it "
        "from that. Reported once."
    )


def is_psf_fit(model: str | None) -> bool:
    """True if a FittedSpectrum came from the PSF-convolved lineshape.

    Takes the `model` string a fit carries (the fit_kind tag, e.g.
    '2lorentzian_x_psf_window_linear'), not a config model name. Scans saved
    before the 2026-08-20 rename carry 'pixel_response' tags — those strings
    live in stored data and are recognised too.
    """
    tag = str(model or "")
    return "lorentzian_x_psf" in tag or "pixel_response" in tag


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
    centers = sorted(float(c) for c in centers)
    if len(centers) == 1:
        return [np.ones_like(x, dtype=bool)]
    # Boundaries at the midpoints between adjacent centers: one segment
    # per peak.
    bounds = [0.5 * (a + b) for a, b in zip(centers[:-1], centers[1:])]
    masks = []
    lo = None
    for b in bounds:
        masks.append((x <= b) if lo is None else ((x > lo) & (x <= b)))
        lo = b
    masks.append(x > lo)
    return masks


def _make_background(background: str, px_fit, centers, offset0, use_window,
                     reflection_bg=None):
    """Return (func(x, *bg_params), p0, lo, hi, n_params).

    The baseline SCOPE follows the fit scope (decision 2026-08-20): a windowed
    fit (use_window, +-beta*width around each peak) gets one baseline segment
    PER PEAK — each segment describes the local gradient under its own peak,
    which is what removes the L-R split; a global fit (no window) gets ONE
    shared baseline over the whole domain. Segment membership is recomputed
    from x on every call, so the same function evaluates on the fit window,
    the full pixel axis and the refined grid. reflection_bg is the (px, R) pair
    for the 'reflection' background.
    """
    def segments(x):
        if use_window:
            return _background_masks(x, centers)
        return [np.ones_like(np.asarray(x, dtype=float), dtype=bool)]

    n_parts = len(segments(px_fit))

    if background == "flat":
        # Constant offset per segment, no slope: the width-safe baseline (a
        # freed slope is odd-symmetric and leaks into the fitted width via
        # covariance — see BACKGROUNDS in find_peaks_config). Lower bound 0:
        # the sline is clipped to >= 0, so a negative background is not physical.
        def func(x, *params):
            x = np.asarray(x, dtype=float)
            out = np.zeros_like(x)
            for i, m in enumerate(segments(x)):
                out = out + m * params[i]
            return out

        return (func, [offset0] * n_parts, [0.0] * n_parts,
                [np.inf] * n_parts, n_parts)

    if background == "linear":
        # Constant + slope per segment. Reference points are fixed from the
        # fit domain so the parameters keep a stable meaning.
        x0 = float(np.mean(px_fit))
        fit_masks = segments(px_fit)
        x0s = [float(np.mean(px_fit[m])) if np.any(m) else x0 for m in fit_masks]

        def func(x, *params):
            x = np.asarray(x, dtype=float)
            out = np.zeros_like(x)
            for i, m in enumerate(segments(x)):
                out = out + m * (params[2 * i] + params[2 * i + 1] * (x - x0s[i]))
            return out

        n = 2 * n_parts
        p0 = []
        for _ in range(n_parts):
            p0 += [offset0, 0.0]
        return func, p0, [-np.inf] * n, [np.inf] * n, n

    if background == "reflection":
        # Per-peak flat offset + ONE shared scale of the measured reflection
        # background (the bg19 minimal model, validated 2026-08-19). The
        # shaped basis function replaces prm1's free slope. Two deliberate
        # restrictions, both measured 2026-08-19/20:
        #   * NO shift parameter — a fitted template shift trades against the
        #     AS centre at ~5 MHz/px; registration belongs to the calibration
        #     (ReflectionBackgroundMapper).
        #   * NO per-peak scale — freeing s per peak removes the S-side
        #     constraint on the scale and re-opens the amplitude<->centre
        #     trade (splits +3..+4 MHz on wide glycerol). Envelope changes
        #     after a realignment are an instrument-state property: verify or
        #     retake the TEMPLATE, do not free the fit.
        if reflection_bg is None:
            raise ValueError(
                "Background 'reflection' needs the mapped reflection "
                "background: pass reflection_background to fit() — build it "
                "with ReflectionBackgroundMapper(...).render(px) (see "
                "spectrum_fitting/reflection_background.py)."
            )
        px_ref, r_ref = reflection_bg

        def func(x, *params):
            x = np.asarray(x, dtype=float)
            r = np.interp(x, px_ref, r_ref, left=0.0, right=0.0)
            out = params[n_parts] * r
            for i, m in enumerate(segments(x)):
                out = out + m * params[i]
            return out

        # s is the sample's pattern strength relative to the reflection-plane
        # template: measured 1e-3..9e-3 across the validated datasets, so 1.0
        # is a generous sanity ceiling, not a tuning.
        n = n_parts + 1
        p0 = [offset0] * n_parts + [1e-3]
        lo = [0.0] * n_parts + [0.0]
        hi = [np.inf] * n_parts + [1.0]
        return func, p0, lo, hi, n

    raise ValueError(f"Unknown background '{background}'.")


class SpectrumFitter:
    def __init__(self):
        # The PSF kernel working values ride inside sline_config (the
        # [global] section) — one fitting config, shared by sample and
        # reference fits (one camera, one kernel; per-section constants
        # would allow the centre-convention mismatch the model guard
        # exists for).
        self.sline_config: SlineFromFrameConfig = sline_from_frame_config.get()
        self.sample_config: FindPeaksConfig = find_peaks_sample_config.get()
        self.reference_config: FindPeaksConfig = find_peaks_reference_config.get()
        # Rows chosen by the automatic band selection, frozen after the first
        # use so the band cannot drift between a scan's calibration and its
        # samples (a one-row difference biases the two peaks by ~3-4 MHz in
        # opposite directions).
        self._auto_rows: list[int] | None = None
        # One-shot flag: the "n_peaks=4 on two-peak data" error is reported
        # once per fitter, not once per frame.
        self._warned_missing_outer = False

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
        self._warned_missing_outer = False

    def auto_select_rows(self, frames) -> list[int]:
        """Locate and freeze the row band from a frame or a stack of frames.

        Call this ONCE per scan with a representative stack (more frames = a
        better-determined centroid), then use the same fitter — or the same
        rows — for that scan's calibration and sample frames. The chosen rows
        are returned so they can be stored with the data.
        """
        rows = select_rows(frames, self.sline_config.n_rows)
        self._auto_rows = rows
        log.info(f"[SpectrumFitter] auto row band: {rows[0]}-{rows[-1]} "
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
            log.warning("[SpectrumFitter] Invalid or empty row list — using full image height.")
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
        """An un-attempted fit (live fitting off, or fit() raised upstream).

        Same shape as a failed fit; the empty model tag ('') means 'never
        fitted', a non-empty tag on an unsuccessful result means 'attempted
        with this recipe and failed'.
        """
        return self._failed_fit(px, sline, model="")

    def _failed_fit(self, px, sline, model: str) -> FittedSpectrum:
        """The ONE failure shape, used by every unsuccessful path.

        A failed fit is a legitimate outcome (no peaks, pure background, the
        microscope moving), so it carries the same record a success does:
        the raw data, the attempted recipe tag, and the row band. All peak
        fields stay None — downstream code keys on is_success.
        """
        return FittedSpectrum(
            is_success=False,
            x_pixels=px,
            sline=sline,
            model=model,
            sline_rows=self._rows_or_none(),
        )

    @staticmethod
    def _fit_kind(n_peaks: int, model: str, use_window: bool,
                  background: str) -> str:
        """The recipe tag a fit result carries (e.g. '2lorentzian_x_psf_window_linear')."""
        kind = f"{n_peaks}{model}"
        if use_window:
            kind += "_window"
        if background != "flat":
            kind += f"_{background}"
        return kind

    def _rows_or_none(self) -> list[int] | None:
        """The row band this fitter sums, for recording on fit results.

        None only when it cannot be known yet (auto mode, band not located).
        """
        try:
            return list(self.get_selected_rows())
        except Exception:
            return None

    def get_total_sline_value(self, sline) -> float:
        if sline is None:
            return 0.0
        return float(np.sum(sline))

    # -----------------------------
    # Lineshape assembly
    # -----------------------------

    def _peak_model(
        self, model, n_peaks, amp, cen, wid,
        center_ranges, x_span, use_window,
    ):
        """Build the peak part of the model.

        Returns (func(x, *peak_params), p0, lo, hi, n_params_per_peak).
        Peaks are always ordered left-to-right, so per-peak options (the
        pixel-response tails, the per-peak baseline) line up with them.
        """
        def width_bounds(i, floor=1e-12):
            if not use_window:
                return floor, x_span / 2
            lo_w = max(1e-6, 0.25 * float(wid[i]))
            return lo_w, max(lo_w * 2, 4.0 * float(wid[i]))

        if model in ("lorentzian", "lorentzian_x_psf"):
            if model == "lorentzian_x_psf":
                sigma = float(self.sline_config.psf_sigma_px)
                tau_l = float(self.sline_config.psf_tau_left_px)
                tau_r = float(self.sline_config.psf_tau_right_px)
                if sigma <= 0.0 and tau_l <= 0.0 and tau_r <= 0.0:
                    raise ValueError(
                        "Model 'lorentzian_x_psf' requires the frozen camera "
                        "constants: set psf_sigma_px and/or psf_tau_left_px / "
                        "psf_tau_right_px in the [global] section of the "
                        "find-peaks config. Measured "
                        "2026-07: 0.25 / 0.40 / 0.20 px (record in "
                        "ccd_characteristics [psf]). With all three at 0 "
                        "this model is just 'lorentzian'."
                    )
                # Per-peak tails by left-to-right POSITION (the tail is a
                # position property of the readout, falling toward higher
                # px — measured 2026-08-20 on the outer cal lines):
                #   2 peaks: [tau_left, tau_right] (the main pair)
                #   4 peaks: [outer_left, tau_left, tau_right, outer_right]
                if n_peaks == 4:
                    taus = [float(self.sline_config.psf_tau_outer_left_px),
                            tau_l, tau_r,
                            float(self.sline_config.psf_tau_outer_right_px)]
                else:
                    taus = [tau_l, tau_r]

                def peak(x, a, c, w, i):
                    return psf_profile(x, a, c, w, sigma, taus[i])
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

            def func(x, *params):
                out = peak(x, params[0], params[1], params[2], 0)
                for i in range(1, n_peaks):
                    out = out + peak(x, params[3 * i], params[3 * i + 1],
                                     params[3 * i + 2], i)
                return out

            p0, lo, hi = [], [], []
            for i in range(n_peaks):
                lo_w, hi_w = width_bounds(i)
                p0 += [amp[i], cen[i], wid[i]]
                lo += [0, center_ranges[i][0], lo_w]
                hi += [np.inf, center_ranges[i][1], hi_w]
            return func, p0, lo, hi, 3

        raise ValueError(f"Unknown model: '{model}'.")

    def fit(
        self,
        px: np.ndarray,
        sline: np.ndarray,
        is_reference_mode: bool,
        reflection_background: np.ndarray | None = None,
        n_peaks: int | None = None,
    ) -> FittedSpectrum:
        """Fit the sline. reflection_background is the reflection background
        mapped onto this px axis (ReflectionBackgroundMapper.render(px));
        required by (and only used with) background='reflection'
        (the 'prmr' preset).

        n_peaks comes from the GLOBAL config (sline_config.n_peaks — one
        ROI, one peak count, shared by sample and reference fits; the
        four-peak standard since 2026-08-21); the argument overrides it
        for analysis scripts and tests. n_peaks=4 jointly fits all four
        VIPA orders — each peak identical (amplitude/centre/width free)
        through the frozen kernel with its own per-position tau; the
        reported left/right peaks stay the INNER main pair, so every
        downstream consumer is unchanged, and the outer pair is reported in
        the outer_* fields.

        Degraded outcomes never raise mid-scan: a frame with no findable
        peaks returns is_success=False (with the raw data, the recipe tag
        and the row band); a 2-peak request that finds one blob — the peaks
        MERGED at small shift — fits that single peak and reports it as two
        coincident half-amplitude peaks with inter_peak_distance 0, tagged
        '1<model>...' (see _build_result)."""
        config = self.reference_config if is_reference_mode else self.sample_config
        # The one resolution path for model / background / window / beta —
        # presets and legacy names included (callers that assign config
        # fields directly bypass FindPeaksConfig.__post_init__, so fit()
        # resolves again through the same function).
        opts = resolve_fit_options(config)
        requested_model = opts.model
        background = opts.background
        use_window = opts.use_window
        beta = opts.beta

        if requested_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model: '{requested_model}'. "
                f"Supported models are {', '.join(SUPPORTED_MODELS)}. "
                f"Windowing and the baseline are separate config options "
                f"(use_window, background), not part of the model name."
            )

        # The model-mixing trap: 'lorentzian_x_psf' defines the peak centre as
        # the Lorentzian core BEFORE the asymmetric tail, ~0.27 px away from a
        # plain Lorentzian's apparent centre. Fitting samples with one
        # convention against a calibration fitted with the other injects a
        # -168 MHz left-right split (measured 2026-08). Calibration and
        # samples must therefore use the same lineshape family.
        if not is_reference_mode:
            reference_model = resolve_fit_options(self.reference_config).model
            psf = "lorentzian_x_psf"
            if (psf in (requested_model, reference_model)
                    and requested_model != reference_model):
                raise ValueError(
                    f"Model mixing: sample model '{requested_model}' with "
                    f"reference model '{reference_model}'. The PSF-convolved "
                    f"centre convention differs from a plain Lorentzian's by "
                    f"~0.27 px (-168 MHz split when mixed), so calibration and "
                    f"samples must both use 'lorentzian_x_psf' (e.g. "
                    f"prm0/prm1) or neither."
                )
            # No camera-constant mismatch check needed: the pr_* constants are
            # global (sline_config.psf_*), so sample and reference fits share one
            # kernel by construction.

        px = np.asarray(px, dtype=np.float64)
        sline = np.asarray(sline, dtype=np.float64)

        if background == "reflection" and reflection_background is None:
            # NO fallback template (user decision 2026-08-24: a
            # stale-alignment default is worse than no correction).
            _warn_missing_reflection_background()
            background = "flat"
        if background == "reflection":
            reflection_background = np.asarray(reflection_background,
                                             dtype=np.float64)
            if reflection_background.shape != px.shape:
                raise ValueError(
                    f"reflection_background must be sampled on the same pixel "
                    f"axis as the sline (got {reflection_background.shape} vs "
                    f"px {px.shape})."
                )

        finite_mask = np.isfinite(px) & np.isfinite(sline)
        px = px[finite_mask]
        sline = sline[finite_mask]
        if reflection_background is not None:
            reflection_background = reflection_background[finite_mask]

        # Keep this if your peak finder expects non-negative data. Remove if you want
        # the offset/background model to handle negative baseline excursions.
        sline = np.clip(sline, 0, None)

        n_requested = (n_peaks if n_peaks is not None
                       else int(self.sline_config.n_peaks))
        if n_requested not in (2, 4):
            raise ValueError(f"n_peaks must be 2 or 4, got {n_requested!r}.")

        pk_ind, pk_info = find_peak_locations(sline, config=config)
        if len(pk_ind) < 1:
            return self._failed_fit(px, sline, self._fit_kind(
                n_requested, requested_model, use_window, background))

        # Selection by amplitude ranking. n_peaks=2 keeps the two strongest,
        # which are always the inner main pair (VIPA side orders are
        # dimmer); n_peaks=4 keeps the outer orders as well.
        pk_ind, pk_info = select_top_n_peaks(pk_ind, pk_info, n_requested)
        amp, cen, wid = self._extract_peak_params(pk_ind, pk_info, px, sline)

        # n_found is how many peaks the finder actually delivered; it may be
        # fewer than n_requested. n_found == 1 with a 2-peak request is the
        # MERGED-PAIR case: at small shifts the two Brillouin peaks overlap
        # into one blob the finder cannot separate. That is legitimate data,
        # so the fit proceeds with a single peak and _build_result reports it
        # as two coincident half-amplitude peaks (see there); the recipe tag
        # then starts with '1', keeping the case visible downstream.
        n_found = len(cen)
        if n_found < 1 or (n_requested == 4 and n_found < 4):
            # A 4-peak fit needs all four orders in view; fewer found means
            # the ROI/thresholds don't support it — fail loudly, no silent
            # fallback to a different model layout.
            if n_requested == 4 and 1 <= n_found < 4:
                # Once per fitter, not per frame: on a two-peak ROI every
                # frame would repeat it; the failed fits carry the record.
                if not self._warned_missing_outer:
                    self._warned_missing_outer = True
                    log.error(f"[SpectrumFitter] n_peaks=4 but only "
                              f"{n_found} peak(s) detected — this data "
                              f"was likely recorded with a two-peak ROI "
                              f"(or the thresholds miss the outer "
                              f"orders). Fits fail until n_peaks is set "
                              f"to 2 in the global fitting config. "
                              f"Reported once; further frames fail "
                              f"silently.")
            return self._failed_fit(px, sline, self._fit_kind(
                n_requested, requested_model, use_window, background))

        # Peaks are ordered left-to-right from here on: the per-peak
        # PSF tails and the per-peak baseline segments rely on it
        # (select_top_n_peaks returns them in height order).
        order = np.argsort(np.asarray(cen, dtype=float))
        amp, cen, wid = (np.asarray(amp)[order], np.asarray(cen)[order],
                         np.asarray(wid)[order])

        x_min = float(np.min(px))
        x_max = float(np.max(px))
        x_span = max(x_max - x_min, 1.0)
        offset0 = float(np.amin(sline))

        if use_window:
            mask = self._build_window_mask(px, cen, wid, beta=beta)
            center_ranges = self._bounded_center_ranges(px, cen, wid, beta=beta)
        else:
            mask = np.ones_like(px, dtype=bool)
            center_ranges = [(x_min, x_max)] * n_found

        px_fit = px[mask]
        sline_fit = sline[mask]

        peak_func, p0_pk, lo_pk, hi_pk, n_per_peak = self._peak_model(
            requested_model, n_found, amp, cen, wid,
            center_ranges, x_span, use_window,
        )
        bg_func, p0_bg, lo_bg, hi_bg, n_bg = _make_background(
            background, px_fit, cen, offset0, use_window,
            reflection_bg=(None if reflection_background is None
                         else (px, reflection_background)),
        )

        n_pk = len(p0_pk)

        def model_func(x, *params):
            return peak_func(x, *params[:n_pk]) + bg_func(x, *params[n_pk:])

        p0 = list(p0_pk) + list(p0_bg)
        bounds = (list(lo_pk) + list(lo_bg), list(hi_pk) + list(hi_bg))

        # Tagged with the number of peaks actually FITTED, so a merged pair
        # (n_found = 1 on a 2-peak request) is recognisable by its '1' prefix.
        fit_kind = self._fit_kind(n_found, requested_model, use_window,
                                  background)

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
            log.warning(f"[SpectrumFitter] Fit failed: {e}")
            return self._failed_fit(px, sline, fit_kind)

        peak_params = [list(popt[i * n_per_peak:(i + 1) * n_per_peak])
                       for i in range(n_found)]
        bg_params = list(popt[n_pk:])

        fit_centers = [p[1] for p in peak_params]
        if any(b < a for a, b in zip(fit_centers, fit_centers[1:])):
            # Re-sort left-to-right and permute the per-peak baseline
            # segments with their peaks (only when the baseline IS per-peak,
            # i.e. windowed; the shared reflection scale, the last parameter,
            # stays put).
            perm = list(np.argsort(fit_centers))
            peak_params = [peak_params[i] for i in perm]
            if use_window and background in ("flat", "linear"):
                k = n_bg // n_found
                groups = [bg_params[i * k:(i + 1) * k] for i in range(n_found)]
                bg_params = [v for i in perm for v in groups[i]]
            elif use_window and background == "reflection":
                offs = bg_params[:-1]
                bg_params = [offs[i] for i in perm] + bg_params[-1:]

        centers = [p[1] for p in peak_params]
        bg_at_peaks = np.atleast_1d(
            bg_func(np.asarray(centers, dtype=float), *bg_params))
        offset_value = float(np.mean(bg_at_peaks))

        return self._build_result(
            px=px,
            sline=sline,
            model_func=model_func,
            popt=popt,
            model=fit_kind,
            mask=mask,
            peak_params=peak_params,
            offset_value=offset_value,
            bg_at_peaks=bg_at_peaks,
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
                      mask: np.ndarray, peak_params, offset_value: float,
                      bg_at_peaks) -> FittedSpectrum:
        """Assemble the result from peaks already parsed left-to-right.

        peak_params[i] is that peak's parameter list; the first three entries
        are always (amplitude, centre, width) for every lineshape.
        bg_at_peaks[i] is the fitted background level at that peak's centre.
        """
        fitted = model_func(px, *popt)
        x_fit, y_fit = refine_fitted_spectrum(model_func, px, popt, factor=10)

        # The REPORTED left/right peaks are always the main pair: with a
        # 4-peak fit that is the INNER pair (positions 2 and 3 of the
        # left-to-right ordering — also the two brightest), so every
        # downstream consumer is unchanged. The outer orders go into the
        # outer_* fields.
        four_peaks = len(peak_params) == 4
        if four_peaks:
            outer_l, left, right, outer_r = peak_params
            bg_left, bg_right = bg_at_peaks[1], bg_at_peaks[2]
        else:
            left = peak_params[0]
            right = peak_params[-1]
            outer_l = outer_r = None
            bg_left, bg_right = bg_at_peaks[0], bg_at_peaks[-1]
        # ONE fitted peak = the merged pair (a 2-peak request whose two
        # Brillouin peaks overlap into a single blob at small shift). It is
        # reported as two COINCIDENT peaks at HALF the fitted amplitude
        # each: the model is linear in amplitude, so two same-width peaks at
        # amp/2 on the same centre sum exactly to the fitted blob — per-peak
        # areas (and the photon counts derived from them) stay additive.
        # inter_peak_distance is 0 by construction, and the model tag's '1'
        # prefix marks the case for downstream consumers.
        two_peaks = len(peak_params) >= 2

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
            left_peak_bg_counts=float(bg_left),
            right_peak_bg_counts=float(bg_right),
            outer_left_peak_center_px=float(outer_l[1]) if four_peaks else None,
            outer_left_peak_width_px=float(outer_l[2]) if four_peaks else None,
            outer_left_peak_amplitude=float(outer_l[0]) if four_peaks else None,
            outer_right_peak_center_px=float(outer_r[1]) if four_peaks else None,
            outer_right_peak_width_px=float(outer_r[2]) if four_peaks else None,
            outer_right_peak_amplitude=float(outer_r[0]) if four_peaks else None,
            outer_left_peak_bg_counts=float(bg_at_peaks[0]) if four_peaks else None,
            outer_right_peak_bg_counts=float(bg_at_peaks[-1]) if four_peaks else None,
            sline_rows=self._rows_or_none(),
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
