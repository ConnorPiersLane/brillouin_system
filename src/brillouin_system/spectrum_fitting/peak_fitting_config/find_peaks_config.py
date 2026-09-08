# config/find_peaks_config.py
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.configs import CONFIG_DIR
from brillouin_system.helpers.thread_safe_config import LazyThreadSafeConfig, ThreadSafeConfig

# The model name selects the LINESHAPE only. Windowing (use_window) and the
# baseline (background) are independent toggles that apply to any lineshape.
# 'dho_x_psf' (2026-08-28): eq.-S2 damped-harmonic-oscillator core (Bailey
# et al., Sci. Adv. 2020) built in each peak's own calibration frequency
# track and convolved with the instrument Lorentzian + camera kernel. The
# fitted center is the acoustic RESONANCE (damping-corrected shift through
# the standard chain) and the fitted width is the ACOUSTIC HWHM directly —
# meant for wide-linewidth (viscous) samples where the DHO center offset
# (~Gamma^2/nu_B-scaled) matters. Sample-only, n_peaks = 2 only, and fits
# need dho_axes from the scan's calibration (see spectrum_fitting/dho.py).
FITTING_MODELS_SAMPLE = [
    "lorentzian",
    "lorentzian_x_psf",
    "dho_x_psf",
    "prm0",
    "prm1",
    "prmr",
]
# One PSF entry, on purpose: the calibration has a single lineshape, and
# offering prm0/prm1 here would be two names for it that silently differ in
# baseline, window and beta. The presets are SAMPLE recipes — their per-peak
# baseline was validated against the local gradient under sample peaks, never
# on calibration frames. Pair a prm sample fit with plain "lorentzian_x_psf";
# the mixing guard normalizes preset names to their lineshape, so that is all
# it asks for. To give the calibration a different baseline deliberately, set
# `background` in the [reference] config.
FITTING_MODELS_REFERENCE = [
    "lorentzian",
    "lorentzian_x_psf",
]

# NOTE on the name: the camera PSF in "lorentzian_x_psf" (sigma + readout
# tails, convolved IN the fit) is not the VIPA instrument response — that one
# is measured by the calibration sidebands and subtracted from the fitted
# width in SpectrumAnalyzer.

# Preset names for the validated production recipes (2026-08-03 decision).
# A preset pins lineshape + background + window + beta together, because the
# combination is what was validated — the width recipe is only valid at
# beta = 3.0 (the halo appears in the wings from ~3.5 gamma out). Use the
# long-form (fitting_model + background) to deviate deliberately.
#   prm0  lorentzian_x_psf + per-peak flat offset — unbiased WIDTHS; single-
#         peak centers carry the halo's odd moment (distance is unaffected).
#   prm1  lorentzian_x_psf + per-peak linear bg   — unbiased CENTERS (the
#         slope absorbs the halo's odd moment) at the cost of a state-
#         dependent width-gap fabrication (up to ~+2.5 MHz, gold session).
# Production usage: fit with prm1; the prm0 companion width-gap difference is
# the QA indicator (accept widths when |gap_prm1 - gap_prm0| <= ~1.5 MHz).
#   prmr  lorentzian_x_psf + per-peak flat offset + ONE shared scale of the
#         MEASURED reflection background ("ReflectionBG", 2026-08-19) — the
#         background model that replaces prm1's linear slope with the
#         instrument's actual stray pattern, registered onto the session
#         through the calibrations (frequency-anchored, so it survives VIPA
#         realignment). Needs reflection_background passed to fit() — see
#         spectrum_fitting/reflection_background.py. Closure-validated
#         2026-08-19/20 (raw template 9/12, calibration-mapped 10/12 cells
#         |split| < 2 MHz at beta 3+4, no per-dataset beta rule; a per-peak
#         scale was tested and REJECTED, splits +3..+4 MHz on wide glycerol).
MODEL_PRESETS = {
    "prm0": {"fitting_model": "lorentzian_x_psf", "background": "flat",
             "use_window": True, "beta": 3.0},
    "prm1": {"fitting_model": "lorentzian_x_psf", "background": "linear",
             "use_window": True, "beta": 3.0},
    "prmr": {"fitting_model": "lorentzian_x_psf",
             "background": "reflection",
             "use_window": True, "beta": 3.0},
}

# Baseline under the peaks. The SCOPE follows the fit scope (2026-08-20
# simplification): a windowed fit (use_window, +-beta*width around each peak)
# gets one baseline segment PER PEAK; a global fit (no window) gets ONE shared
# baseline. The old flat_per_peak/linear_per_peak names are normalised to
# flat/linear — with the window on (how they were always used) the behaviour
# is identical.
#   flat        constant offset (per peak when windowed) — the width-safe
#               baseline (a free slope is odd-symmetric: it corrects the
#               CENTER exactly but leaks into the WIDTH via covariance,
#               fabricating up to ~+2.5 MHz of L-R width gap; closure-tested
#               2026-08-02).
#   linear      constant + slope (per peak when windowed) — the one that
#               removes the L-R split in water (measured 2026-07:
#               -3.3 -> -0.4 MHz), because the bias comes from the LOCAL
#               gradient under each peak.
#   reflection  per-peak constant offsets, plus ONE shared scale of the
#               MEASURED reflection-plane background (the laser's satellite
#               comb imaged by the VIPA — the structure the linear slope was
#               absorbing). The template is registered onto the session's
#               pixel axis through the calibrations; fits then need
#               reflection_background (see
#               spectrum_fitting/reflection_background.py). One parameter
#               fewer than linear, a shaped basis instead of a free slope;
#               deliberately NO shift parameter (a fitted shift trades
#               against the AS centre at ~5 MHz/px) and NO per-peak scale
#               (re-opens the amplitude<->centre trade, +3..+4 MHz on wide
#               glycerol; both measured 2026-08-19/20).
#   flat_shared ONE constant offset for the whole fit, even when windowed —
#               the CALIBRATION (reference) baseline since 2026-08-25. A
#               calibration frame is two bright sidebands on one dark
#               level; giving each side its own offset lets the offset
#               trade against that side's peak WIDTH (opposite directions
#               per side, worst at the 8 GHz sweep edge where the
#               sidebands crowd: -0.036/+0.020 px), which bends the
#               degree-2 width-vs-pixel calibration polynomial and
#               fabricates ~+10 MHz of AS-S linewidth gap at the sample
#               position through the at-position subtraction. Validated
#               against the Figure 4(d) workbook chain (shared-offset cal
#               fits: widths on Holmes, eta_b/eta_s 2.79 vs 2.78, AS-S
#               gaps ~3 MHz). Peak CENTRES are insensitive to this choice
#               — shifts and splits do not move.
# Costs ~10% in single-frame distance precision. Validated on liquids only:
# cornea splits are already unbiased and get WORSE with a background term.
# reflection_per_peak: EXPERIMENTAL QC variant — per-peak offsets plus a
# SEPARATE template scale per peak. Rejected for production 2026-08-20
# (freeing s per side re-opens the amplitude<->centre trade, splits
# +3..+4 MHz on wide glycerol); kept selectable via the long-form config
# for side-by-side tests only. Never put it in a preset.
BACKGROUNDS = ["flat", "linear", "reflection", "reflection_per_peak",
               "flat_shared"]

# Legacy background names (removed 2026-08-20): the per-peak variants are now
# simply flat/linear under a windowed fit.
_LEGACY_BACKGROUNDS = {
    "flat_per_peak": "flat",
    "linear_per_peak": "linear",
}

# Collection weight W(v) over the NA cone, for the post-hoc scalar correction
# (na_lineshape.na_mean_shift_ratio) — fit as at low NA, then divide the
# measured shift by <cos(v/2)>; the correction never enters the fit:
#   none              no NA correction: the ratio is exactly 1.0 and the
#                     other na_* fields are ignored. An EXPLICIT choice —
#                     the default stays "uniform", so an unconfigured NA
#                     (na_collection = 0) still fails loudly when a
#                     correction is requested rather than silently
#                     un-correcting.
#   uniform           W = sin(v): hard pupil, uniform transmission. The
#                     parameter-free low-NA model (paper Fig. 4) — at NA 0.14
#                     it is worth about +3.5 MHz on water.
#   uniform_gaussian  W = sin(v) * exp(-2 (v/v0)^2): hard clip plus the
#                     Gaussian fiber-coupling apodization (paper Fig. 5) —
#                     required at NA 0.42, where a uniform pupil overcorrects.
#                     v0 comes from na_beam_diameter_mm (the per-session knob,
#                     calibrated on water) and na_focal_length_mm.
NA_WEIGHTINGS = ["none", "uniform", "uniform_gaussian"]

# Models that were removed, with the migration hint shown if one is still set.
_REMOVED_MODELS = {
    "asym_lorentzian": "tested on calibration data and rejected (it triples the "
                       "residual sine); use 'lorentzian_x_psf' for an "
                       "asymmetric instrument lineshape",
    "lorentzian_quad_bg": "never implemented; use background='linear' "
                          "(a quadratic baseline was tested and is degenerate "
                          "with the Lorentzian wings)",
    "voigt": "removed 2026-08-20 (unused; the DHO/voigt lineshape question "
             "was answered — estimator choice, not physics); use "
             "'lorentzian_x_psf'",
    "na_lorentzian": "NA lineshape models removed 2026-08-20; the high-NA "
                     "recipe is the POST-HOC scalar correction (na_weighting "
                     "config + na_lineshape.na_mean_shift_ratio) applied to a "
                     "standard prm fit",
    "na_gauss_lorentzian": "NA lineshape models removed 2026-08-20; use the "
                           "post-hoc correction with na_weighting = "
                           "'uniform_gaussian'",
    "pixel_response": "renamed 2026-08-20 to 'lorentzian_x_psf' (same "
                      "lineshape: Lorentzian convolved with the frozen "
                      "camera PSF)",
}

# Legacy model names that folded a baseline choice into the lineshape name.
_LEGACY_BACKGROUND_MODELS = {
    "lorentzian_linear_bg": ("lorentzian", "linear"),
}


@dataclass(frozen=True)
class ResolvedFitOptions:
    """What a fit will actually run, after preset and legacy-name resolution."""
    model: str
    background: str
    use_window: bool
    beta: float


def resolve_fit_options(config) -> ResolvedFitOptions:
    """THE resolution path for a fitting config: legacy '<model>_window'
    names, legacy background names, legacy background-in-the-model names,
    and the prm* presets, applied in that order.

    FindPeaksConfig.__post_init__ normalises through here on construction,
    but callers that assign config fields directly bypass it — so everything
    that needs the effective (model, background, use_window, beta) resolves
    through this one function instead of re-implementing the rules.
    """
    model = str(getattr(config, "fitting_model", ""))
    use_window = bool(getattr(config, "use_window", True))
    background = str(getattr(config, "background", "flat"))
    beta = float(getattr(config, "beta", 4.0))
    if model.endswith("_window"):
        model = model[: -len("_window")]
        use_window = True
    if model in _LEGACY_BACKGROUND_MODELS:
        model, background = _LEGACY_BACKGROUND_MODELS[model]
    background = _LEGACY_BACKGROUNDS.get(background, background)
    if model in MODEL_PRESETS:
        preset = MODEL_PRESETS[model]
        model = preset["fitting_model"]
        background = preset["background"]
        use_window = preset["use_window"]
        beta = preset["beta"]
    return ResolvedFitOptions(model=model, background=background,
                              use_window=use_window, beta=beta)


# The camera PSF working values live in the [global] section below
# (SlineFromFrameConfig) — ONE fitting config, no nested sub-config (user
# decision 2026-08-20: "a config in a config is not a good design"). The
# MEASURED kernel record (values + date + method) is
# psf_measurement.PSF_MEASURED in this package, next to
# measure_psf_kernel.py (the PSF is peaks domain, not camera — user rule
# 2026-08-24); the GUI shows it in brackets and never writes it.


@dataclass
class FindPeaksConfig:
    prominence_fraction: float
    min_peak_width: int
    min_peak_height: int
    rel_height: float
    wlen_pixels: int
    fitting_model: str
    # Fit only within +-beta*width around each peak instead of the whole sline.
    # Replaces the old '<model>_window' names, which are still accepted as input
    # and normalised here.
    use_window: bool = True
    # Baseline model, independent of the lineshape — see BACKGROUNDS above.
    background: str = "flat"
    beta: float = 4.0
    # REFERENCE fits only: how the calibration-line centres (the px->GHz
    # polynomials) are obtained.
    #   "parametric": one Lorentzian(g) x Gauss(sigma) x tail(tau) x pixel
    #     fit per frame — the centre-validated chain (sine ~0.1 MHz).
    #   "template": no instrument model at all — plain-Lorentzian first
    #     guess, stacking on frequency-smoothed centres, template refit
    #     (spectrum_fitting/template_calibration.py; 2026-09-07, sine
    #     0.05-0.13 MHz). The same stack supplies the DHO sample kernels,
    #     so sample and calibration share one centre convention.
    centre_method: str = "parametric"

    def __post_init__(self):
        # All legacy-name and preset rules live in resolve_fit_options —
        # the same path the fitter uses on configs that bypass this method.
        resolved = resolve_fit_options(self)
        if resolved.model in _REMOVED_MODELS:
            raise ValueError(
                f"Fitting model '{resolved.model}' has been removed: "
                f"{_REMOVED_MODELS[resolved.model]}."
            )
        if resolved.background not in BACKGROUNDS:
            raise ValueError(
                f"Unknown background '{resolved.background}'. "
                f"Choose one of {BACKGROUNDS}."
            )
        self.fitting_model = resolved.model
        self.background = resolved.background
        self.use_window = resolved.use_window
        self.beta = resolved.beta
        if self.centre_method not in CENTRE_METHODS:
            raise ValueError(
                f"Unknown centre_method '{self.centre_method}'. "
                f"Choose one of {CENTRE_METHODS}."
            )


@dataclass
class SampleFindPeaksConfig(FindPeaksConfig):
    """Sample-fit config: FindPeaksConfig plus the NA collection model.

    The NA fields describe the collection cone of the SAMPLE illumination and
    drive the post-hoc scalar correction (na_lineshape.na_mean_shift_ratio).
    They have no meaning for reference (calibration) fits — the EOM sidebands
    are elastic light, no cone model — which is why the reference section uses
    the plain FindPeaksConfig.

    # na_weighting: collection weight over the cone — see NA_WEIGHTINGS above.
    #   "uniform" (NA 0.14 recipe): hard pupil only; na_collection is then the
    #     EFFECTIVE NA (it absorbs any apodization).
    #   "uniform_gaussian" (NA 0.42 recipe): na_collection is the NOMINAL
    #     objective NA (physical pupil edge); the apodization is modeled
    #     explicitly via the two geometry fields below.
    # na_collection: hard aperture clip as an NA (alpha = arcsin(NA/n));
    #   0.0 = unset -> the NA routes refuse to run.
    # "uniform_gaussian" only — Gaussian fiber-coupling weight
    # exp(-2 (v/v0)^2), v0 = arcsin(sin(arctan((D/2)/f))/n):
    #   na_beam_diameter_mm: D, 1/e^2 diameter of the collection-fiber mode at
    #     the objective pupil (collimator output beam; F810APC-780 nominal
    #     7.5 mm). The session-calibration knob: tune on water (effective < nominal).
    #   na_focal_length_mm: f, focal length of the OBJECTIVE (20X: 10, 5X: 40).
    # na_n_sample: refractive index of the sample medium.
    """
    na_weighting: str = "uniform"
    na_collection: float = 0.0
    na_beam_diameter_mm: float = 0.0
    na_focal_length_mm: float = 0.0
    na_n_sample: float = 1.33
    # How far beyond the calibrated EOM sweep the reflection-background
    # registration is trusted, per side [GHz] (prmr fits only; outside the
    # trusted range the rendered template is 0). 0.7 = the validated
    # production default. Raise DELIBERATELY for high-shift samples whose
    # peaks sit beyond the sweep — e.g. 2.0 on a 4-8 GHz sweep reaches
    # 10 GHz (plastic at 9.6) — accepting that the quadratic track
    # registrations then extrapolate unverified; see the
    # ReflectionBackgroundMapper docstring for the measured behaviour and
    # caveats (2026-08-25).
    reflection_margin_ghz: float = 0.7
    # Instrument kernel of the 'dho_x_psf' sample model (ignored by every
    # other model and by reference fits):
    #   "parametric": Lorentzian(g_inst from the calibration width
    #     polynomial) (x) Gauss(sigma) (x) ExpTail(tau) (x) pixel — the
    #     centre-validated chain.
    #   "measured": the instrument response stacked from the scan's own
    #     calibration frames at each inner peak's position (no functional
    #     form; spectrum_fitting/measured_kernel.py). Adopted 2026-09-07:
    #     the parametric Stokes kernel carries ~30 % too much wing and read
    #     the DHO acoustic width 3-4 % low; the measured kernel fits the
    #     elastic lines to the noise, equalises the two peaks' widths and
    #     leaves the resonance unchanged. Needs raw calibration frames.
    dho_kernel: str = "parametric"

    def __post_init__(self):
        super().__post_init__()
        if self.dho_kernel not in DHO_KERNELS:
            raise ValueError(
                f"Unknown dho_kernel '{self.dho_kernel}'. "
                f"Choose one of {DHO_KERNELS}."
            )
        if self.na_weighting not in NA_WEIGHTINGS:
            raise ValueError(
                f"Unknown na_weighting '{self.na_weighting}'. "
                f"Choose one of {NA_WEIGHTINGS}."
            )
        if not self.reflection_margin_ghz > 0.0:
            raise ValueError(
                f"reflection_margin_ghz must be positive "
                f"(got {self.reflection_margin_ghz})."
            )


DHO_KERNELS = ["parametric", "measured"]
CENTRE_METHODS = ["parametric", "template"]
ENVELOPE_SOURCES = ["config", "measured"]

ROW_SELECTIONS = ["manual", "auto"]


@dataclass
class SlineFromFrameConfig:
    pixel_offset_left: int
    pixel_offset_right: int
    # Rows summed into the spectral line. With row_selection = "manual" this
    # list is used as given. With "auto" the band is located automatically:
    # n_rows contiguous rows centred on the line's intensity centroid, chosen
    # ONCE and then frozen (see spectrum_fitting/row_selection.py — a
    # calibration-vs-sample band mismatch biases the peaks ~3-4 MHz per row;
    # one shared fitter rules that out).
    selected_rows: list[int]
    row_selection: str = "manual"
    n_rows: int = 13
    # How many VIPA orders to fit: 2 = the inner main pair, 4 = all four
    # orders jointly (each with its own per-position readout tail). GLOBAL,
    # because it is a property of the recorded ROI — one camera frame, one
    # peak count, shared by sample and reference fits. The STANDARD since
    # 2026-08-21 is 4 wherever the ROI contains the outer orders: the
    # calibration then builds a track per order and analyze() reports the
    # per-order shifts plus their inverse-variance combination. On data
    # recorded with a two-peak ROI, n_peaks = 4 stops with an error (the
    # calibration refuses; sample fits fail loudly) — set 2 for that data.
    n_peaks: int = 2
    # Camera PSF working values (the 'lorentzian_x_psf' kernel): Gaussian
    # charge-diffusion blur and the one-sided readout tails, per peak, toward
    # higher pixel numbers. GLOBAL — one camera, one kernel, shared by the
    # sample and reference fits (different kernels would define different
    # peak-centre conventions = the model-mixing artifact). Not fitted per
    # frame. Defaults = the MEASURED kernel; the measurement record
    # (values + date + method) is psf_measurement.PSF_MEASURED in this
    # package, next to measure_psf_kernel.py — re-measure after any
    # camera/ROI change, and update both. The sigma is per peak like the
    # taus (blur is a POSITION property on the sensor; split from the old
    # shared psf_sigma_px). ALL EIGHT constants user-adopted 2026-09-03
    # from the four-peak 4001-point fine-sweep determination of 2026-09-02
    # (four runs, agreement +-0.02 px across a realignment; residual
    # folded sines 0.06-0.09 MHz on outer_left/left/right; record in
    # Data/2026-9-2/determine_fourpeak_summary.txt + phase3 files).
    # One model family, Gauss + one-sided tail (+ measured row-tilt
    # boxcar on the OUTER orders, restored 2026-09-05 for the outer
    # width criterion — it took outer_left's water closure 0.82->0.94+;
    # the inner peaks measurably REJECT the top-hat, their bell-shaped
    # row smear already lives in sigma/tau). With the boxes in, the
    # outer sigma/tau are the PHYSICAL 09-04 determination (four-run
    # agreement; outer_right tau +-0.001 px), not the old effective
    # 0.39/0.95, 0.36/0.0 that impersonated the smear.
    psf_sigma_left_px: float = 0.26
    psf_sigma_right_px: float = 0.27
    psf_tau_left_px: float = 0.39
    psf_tau_right_px: float = 0.17
    psf_sigma_outer_left_px: float = 0.14
    psf_sigma_outer_right_px: float = 0.15
    psf_tau_outer_left_px: float = 0.13
    psf_tau_outer_right_px: float = 0.08
    psf_box_outer_left_px: float = 1.95
    psf_box_outer_right_px: float = 0.85
    # OUTER_RIGHT near-core satellite (RESTORED to production
    # 2026-09-05, user): an intrinsic scaled displaced copy of that
    # order's line (same gamma, same kernel, ratio at delta px),
    # measured 2026-09-02 and blind-validated. Without it the
    # outer_right position wobbles once per pixel by ~3.2 MHz and its
    # four-peak-DHO resonance reads ~-11 MHz low (measured 09-05); no
    # (sigma, tau) can absorb it. Applied to the outer_right peak in
    # BOTH four-peak paths (lorentzian_x_psf and dho_x_psf); ratio 0
    # disables.
    psf_sat_ratio_outer_right: float = 0.037
    psf_sat_delta_outer_right_px: float = -1.23
    # VIPA intensity-envelope gradient per peak [1/px] (MEASURED
    # 2026-09-06, roll-off-free same-frame amplitude ratios,
    # Data/2026-9-2/envelope_slopes_clean.py). The envelope MULTIPLIES
    # the spectrum; its gradient across a peak pulls a symmetric fit by
    # ~k(gamma)*eps*gamma_px^2 toward the envelope top — the outer
    # orders' long-standing shift systematic (predicted vs measured
    # outer-inner offsets: water -12/-9 vs -11/-12, 50wt -67/-46 vs
    # -68/-38, zero fitted parameters). Applied in the FOUR-PEAK fits
    # only (all four peaks, multiplicative exp(eps*(x-cen)) on each
    # peak model); the two-peak paper chain is deliberately untouched
    # (correcting it would move the inner shift convention by ~+5 MHz
    # at water — a PAPER decision, not a default). 0 disables.
    # PRODUCTION: outer slopes only (09-06 verification: env on the
    # INNER peaks inflates their fitted widths +15 MHz and is not
    # needed for their shifts — inner slopes stay 0 in production; the
    # measured inner values (+0.0239/−0.0265) live in the comment and
    # the measurement script for the pending PAPER decision).
    env_slope_outer_left_perpx: float = 0.0302
    env_slope_left_perpx: float = 0.0
    env_slope_right_perpx: float = 0.0
    env_slope_outer_right_perpx: float = -0.0253
    # Where the envelope slopes come from:
    #   "config":   the four constants above (measured on the 9-2 sweeps).
    #   "measured": per scan, from the scan's own calibration frames — the
    #     same-sideband line areas along the sweep give ln g(x_a) - ln g(x_b)
    #     with the drive roll-off cancelled (spectrum_fitting/envelope.py,
    #     validated 2026-09-06: inner slopes stable to 3 % within a state,
    #     but a factor two between alignment states). Needs the four-order
    #     ROI; falls back to the constants when the frames carry two lines.
    envelope_source: str = "config"

    def __post_init__(self):
        if self.envelope_source not in ENVELOPE_SOURCES:
            raise ValueError(
                f"Unknown envelope_source '{self.envelope_source}'. "
                f"Choose one of {ENVELOPE_SOURCES}."
            )
        if self.row_selection not in ROW_SELECTIONS:
            raise ValueError(
                f"Unknown row_selection '{self.row_selection}'. "
                f"Choose one of {ROW_SELECTIONS}."
            )
        if self.n_peaks not in (2, 4):
            raise ValueError(f"n_peaks must be 2 or 4, got {self.n_peaks!r}.")

    def __getattr__(self, name):
        # No aliases (repo rule): the shared sigma was SPLIT per peak
        # 2026-08-31 — old code must be renamed, not silently served one of
        # the two sides. (__getattr__ only fires for missing attributes, so
        # the real fields are unaffected.)
        if name == "psf_sigma_px":
            raise AttributeError(
                "SlineFromFrameConfig.psf_sigma_px was split per peak "
                "(2026-08-31): use psf_sigma_left_px / psf_sigma_right_px."
            )
        if name in ("psf_box_left_px", "psf_box_right_px"):
            raise AttributeError(
                f"SlineFromFrameConfig.{name} does not exist: the INNER "
                "peaks carry no row-tilt boxcar BY MEASUREMENT (their "
                "bell-weighted smear is Gauss+tail-shaped and lives in "
                "sigma/tau; a top-hat there ruins their sines) — only "
                "psf_box_outer_left_px / psf_box_outer_right_px are real."
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

@dataclass
class FittingConfigs:
    sample_config: SampleFindPeaksConfig
    reference_config: FindPeaksConfig
    sline_config: SlineFromFrameConfig

FIND_PEAKS_TOML_PATH = CONFIG_DIR / "find_peaks_config.toml"

# Keys that used to live duplicated in the [sample]/[reference] sections
# (under their OLD pr_* names): the camera constants moved to the global
# [global] section (now psf_*), na_* is sample-only, and n_peaks moved to
# [global] (2026-08-21: a property of the ROI, not of the section). Dropped
# silently when an older TOML still carries them so those files keep
# loading; any other unknown key still raises.
_MOVED_SECTION_KEYS = {
    "pr_sigma_px", "pr_tau_left_px", "pr_tau_right_px",
    "na_weighting", "na_collection", "na_beam_diameter_mm",
    "na_focal_length_mm", "na_n_sample", "n_peaks",
}

def load_config_section(path: Path, section: str) -> FindPeaksConfig:
    cls = SampleFindPeaksConfig if section == "sample" else FindPeaksConfig
    with path.open("rb") as f:
        raw = tomli.load(f)[section]
    names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in raw.items()
              if k in names or k not in _MOVED_SECTION_KEYS}
    return cls(**kwargs)

def load_sline_from_frame_config(path: Path) -> SlineFromFrameConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["global"]
    # Stored-data compatibility only (the field itself has no alias): a TOML
    # from before 2026-08-31 carries ONE shared psf_sigma_px — it applies to
    # both sides. Explicit per-peak keys win.
    if "psf_sigma_px" in raw:
        shared = raw.pop("psf_sigma_px")
        raw.setdefault("psf_sigma_left_px", shared)
        raw.setdefault("psf_sigma_right_px", shared)
    # A TOML from the brief all-boxcar era (2026-09-04) carries INNER
    # box keys; those never survived (the inner peaks reject the
    # top-hat) — drop them. The outer box keys are real fields.
    for key in ("psf_box_left_px", "psf_box_right_px"):
        raw.pop(key, None)
    return SlineFromFrameConfig(**raw)

def save_config_section(path: Path, section: str, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)
    data[section] = asdict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)

# Global configuration instances
find_peaks_sample_config = LazyThreadSafeConfig(lambda: load_config_section(FIND_PEAKS_TOML_PATH, "sample"))
find_peaks_reference_config = LazyThreadSafeConfig(lambda: load_config_section(FIND_PEAKS_TOML_PATH, "reference"))
sline_from_frame_config = LazyThreadSafeConfig(lambda: load_sline_from_frame_config(FIND_PEAKS_TOML_PATH))
