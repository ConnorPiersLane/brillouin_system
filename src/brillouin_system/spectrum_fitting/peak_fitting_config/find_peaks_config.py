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
# Four sample models (2026-09-10): a Lorentzian or a DHO core, bare (pixel
# box only) or through the MEASURED instrument kernel ('_x_psf'). The
# kernel and DHO models fit against the scan's own calibration axes and
# are sample-only; a '_x_psf' fit's width is the sample width.
FITTING_MODELS_SAMPLE = [
    "lorentzian",
    "lorentzian_x_psf",
    "dho",
    "dho_x_psf",
]
# The calibration lines are elastic — their profile IS the instrument
# kernel — so the reference fit is the plain Lorentzian (peak location and
# the live display; the axis itself comes from the template chain).
FITTING_MODELS_REFERENCE = [
    "lorentzian",
]

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
    names, legacy background names and legacy background-in-the-model
    names, applied in that order.

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
    return ResolvedFitOptions(model=model, background=background,
                              use_window=use_window, beta=beta)


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
    def __post_init__(self):
        # All legacy-name and preset rules live in resolve_fit_options —
        # the same path the fitter uses on configs that bypass this method.
        resolved = resolve_fit_options(self)
        if resolved.model not in FITTING_MODELS_SAMPLE:
            raise ValueError(
                f"Unknown fitting_model '{resolved.model}'. Choose one of "
                f"{FITTING_MODELS_SAMPLE} (reference fits: "
                f"{FITTING_MODELS_REFERENCE})."
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
    # registration is trusted, per side [GHz] (reflection-background fits only; outside the
    # trusted range the rendered template is 0). 0.7 = the validated
    # production default. Raise DELIBERATELY for high-shift samples whose
    # peaks sit beyond the sweep — e.g. 2.0 on a 4-8 GHz sweep reaches
    # 10 GHz (plastic at 9.6) — accepting that the quadratic track
    # registrations then extrapolate unverified; see the
    # ReflectionBackgroundMapper docstring for the measured behaviour and
    # caveats (2026-08-25).
    reflection_margin_ghz: float = 0.7
    def __post_init__(self):
        super().__post_init__()
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


ENVELOPE_SOURCES = ["config", "measured"]
# How the measured envelope enters a FOUR-PEAK DHO fit (envelope_source =
# "measured", template chain): the local slope at the found centre,
# exp(g'(c) (x - c)), or the full curve exp(g(x) - g(c)) over the window.
# The two-peak path always uses the slope. Analysis knob (2026-09-09).
ENVELOPE_APPLY = ["slope", "curve"]
# Where the measured DHO sample kernels come from: the scan's own calibration
# (template chain), or a stored node table (Epsf.save) for the lines named by
# kernel_file_lines.
KERNEL_SOURCES = ["scan", "file"]
KERNEL_FILE_LINES = ["outer", "all"]

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
    # "slope" (production) or "curve": see ENVELOPE_APPLY. Only the
    # four-peak DHO path on the template chain reads it.
    envelope_apply: str = "slope"
    # Where the measured DHO sample KERNELS come from (sample model
    # 'dho_x_psf'; the plain Lorentzian reads none):
    #   "scan": the scan's own calibration node table (the 41-point sweep).
    #   "file": a stored node table (Epsf.save, e.g. built on a 401-point
    #     fine sweep) for the lines named by kernel_file_lines; the other
    #     lines, the FREQUENCY AXIS (template centres) and the ENVELOPE
    #     slopes always stay per scan. Motivation (2026-09-09): the standard
    #     41-point sweep steps the outer-left line by 0.54 px per frame, a
    #     half-pixel alias that leaves its stacked profile wiggly and 5 %
    #     too narrow, while the inner pair and outer_right are sampled fine.
    #     A stored table pins the sample kernel's centre CONVENTION to the
    #     file's profile while the axis carries the scan's, so the inner
    #     distance must be checked against kernel_source = "scan" whenever
    #     the file changes (< 0.3 MHz; see spectrum_fitting/epsf.py).
    kernel_source: str = "scan"
    # path of the stored node table (Epsf.save); required for "file"
    kernel_file: str = ""
    # "outer": outer_left + outer_right from the file, inner pair per scan;
    # "all": every fitted line from the file
    kernel_file_lines: str = "outer"
    # With "file" every scan's own calibration profile is compared with the
    # stored one at the sample positions (epsf.FileKernels.check, ~1 ms,
    # always on, logged): a line beyond the limits below falls back to the
    # scan's own kernel with a WARNING (a GUI may ask first,
    # fit_axial_scan.set_kernel_mismatch_handler). The limits (2026-09-09,
    # 200 calibrations over three alignment states): the well-sampled lines
    # agree with the stored table to 0.005 px centre offset, 1 % HWHM and
    # 0.6 % rms; the outer-left alias of a 41-point sweep reaches 0.027 px,
    # 3.2 % and 2.6 %. Centre offset = the shift that lays the scan's
    # profile on the file's (px); it is directly a shift error on that
    # order (~300 MHz/px), hence tighter on the inner pair (user 09-09).
    kernel_match_shift_px_inner: float = 0.01
    kernel_match_shift_px_outer: float = 0.03
    kernel_match_hwhm_fraction: float = 0.06
    kernel_match_rms_percent: float = 4.0

    def __post_init__(self):
        if self.envelope_source not in ENVELOPE_SOURCES:
            raise ValueError(
                f"Unknown envelope_source '{self.envelope_source}'. "
                f"Choose one of {ENVELOPE_SOURCES}."
            )
        if self.envelope_apply not in ENVELOPE_APPLY:
            raise ValueError(
                f"Unknown envelope_apply '{self.envelope_apply}'. "
                f"Choose one of {ENVELOPE_APPLY}."
            )
        if self.kernel_source not in KERNEL_SOURCES:
            raise ValueError(
                f"Unknown kernel_source '{self.kernel_source}'. "
                f"Choose one of {KERNEL_SOURCES}."
            )
        if self.kernel_file_lines not in KERNEL_FILE_LINES:
            raise ValueError(
                f"Unknown kernel_file_lines '{self.kernel_file_lines}'. "
                f"Choose one of {KERNEL_FILE_LINES}."
            )
        self.kernel_file = str(self.kernel_file or "")
        for f in ("kernel_match_shift_px_inner", "kernel_match_shift_px_outer",
                  "kernel_match_hwhm_fraction", "kernel_match_rms_percent"):
            if not float(getattr(self, f)) > 0.0:
                raise ValueError(f"{f} must be > 0, got {getattr(self, f)!r}.")
        if self.kernel_source == "file" and not self.kernel_file.strip():
            raise ValueError(
                "kernel_source = 'file' needs kernel_file (the path of a "
                "stored ePSF node table written by Epsf.save)."
            )
        if self.row_selection not in ROW_SELECTIONS:
            raise ValueError(
                f"Unknown row_selection '{self.row_selection}'. "
                f"Choose one of {ROW_SELECTIONS}."
            )
        if self.n_peaks not in (2, 4):
            raise ValueError(f"n_peaks must be 2 or 4, got {self.n_peaks!r}.")


@dataclass
class FittingConfigs:
    sample_config: SampleFindPeaksConfig
    reference_config: FindPeaksConfig
    sline_config: SlineFromFrameConfig

FIND_PEAKS_TOML_PATH = CONFIG_DIR / "find_peaks_config.toml"

# Keys a TOML must NOT carry any more. They are refused with the reason, never
# dropped silently (user rule 2026-09-10): a stale key means a stale file.
_RETIRED_KEYS = {
    **{k: "the parametric camera PSF was removed 2026-09-10" for k in (
        "psf_sigma_px", "psf_sigma_left_px", "psf_sigma_right_px",
        "psf_tau_left_px", "psf_tau_right_px",
        "psf_sigma_outer_left_px", "psf_sigma_outer_right_px",
        "psf_tau_outer_left_px", "psf_tau_outer_right_px",
        "psf_box_left_px", "psf_box_right_px",
        "psf_box_outer_left_px", "psf_box_outer_right_px",
        "psf_sat_ratio_outer_right", "psf_sat_delta_outer_right_px",
        "pr_sigma_px", "pr_tau_left_px", "pr_tau_right_px",
        "dho_kernel", "centre_method")},
    "kernel_check": "the stored-PSF check is always on since 2026-09-09",
    "n_peaks": "moved to the [global] section 2026-08-21",
    "save_calibration_frames": "removed 2026-08-24, the frames always travel",
}


def _refuse_unknown_keys(raw: dict, names: set, section: str, path: Path):
    unknown = [k for k in raw if k not in names]
    if not unknown:
        return
    reasons = "; ".join(f"'{k}': {_RETIRED_KEYS[k]}" if k in _RETIRED_KEYS
                        else f"'{k}': not a field" for k in unknown)
    raise ValueError(
        f"{path.name} [{section}] carries keys this config does not have — "
        f"{reasons}. Delete them from the file.")


def load_config_section(path: Path, section: str) -> FindPeaksConfig:
    cls = SampleFindPeaksConfig if section == "sample" else FindPeaksConfig
    with path.open("rb") as f:
        raw = tomli.load(f)[section]
    _refuse_unknown_keys(raw, {f.name for f in fields(cls)}, section, path)
    return cls(**raw)


def load_sline_from_frame_config(path: Path) -> SlineFromFrameConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["global"]
    _refuse_unknown_keys(raw, {f.name for f in fields(SlineFromFrameConfig)}, "global", path)
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
