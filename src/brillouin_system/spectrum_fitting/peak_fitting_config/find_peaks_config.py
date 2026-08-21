# config/find_peaks_config.py
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig

# The model name selects the LINESHAPE only. Windowing (use_window) and the
# baseline (background) are independent toggles that apply to any lineshape.
FITTING_MODELS_SAMPLE = [
    "lorentzian",
    "lorentzian_x_psf",
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
# Costs ~10% in single-frame distance precision. Validated on liquids only:
# cornea splits are already unbiased and get WORSE with a background term.
BACKGROUNDS = ["flat", "linear", "reflection"]

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


# The camera PSF constants MOVED to the ccd_characteristics package
# (2026-08-20): they are measured camera properties, not fitting
# configuration — one home for every obtained instrument number, next to
# the scripts that measure them. Re-exported here so existing imports
# (spectrum_fitter, GUI, tests) keep working.
from brillouin_system.ccd_characteristics.ccd_characteristics import (  # noqa: E402
    PsfConstants,
    psf_config,
)


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
        model = str(self.fitting_model)
        # Legacy '<model>_window' names -> base name + use_window.
        if model.endswith("_window"):
            model = model[: -len("_window")]
            self.use_window = True
        # Legacy names that folded the baseline into the lineshape name.
        if model in _LEGACY_BACKGROUND_MODELS:
            model, self.background = _LEGACY_BACKGROUND_MODELS[model]
        # Legacy per-peak background names -> the plain names (the scope now
        # follows use_window, so a windowed fit is per-peak either way).
        if self.background in _LEGACY_BACKGROUNDS:
            self.background = _LEGACY_BACKGROUNDS[self.background]
        # Preset names pin the validated combination (see MODEL_PRESETS):
        # they override background, use_window and beta.
        if model in MODEL_PRESETS:
            preset = MODEL_PRESETS[model]
            model = preset["fitting_model"]
            self.background = preset["background"]
            self.use_window = preset["use_window"]
            self.beta = preset["beta"]
        if model in _REMOVED_MODELS:
            raise ValueError(
                f"Fitting model '{model}' has been removed: "
                f"{_REMOVED_MODELS[model]}."
            )
        if self.background not in BACKGROUNDS:
            raise ValueError(
                f"Unknown background '{self.background}'. "
                f"Choose one of {BACKGROUNDS}."
            )
        self.fitting_model = model


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

    def __post_init__(self):
        super().__post_init__()
        if self.na_weighting not in NA_WEIGHTINGS:
            raise ValueError(
                f"Unknown na_weighting '{self.na_weighting}'. "
                f"Choose one of {NA_WEIGHTINGS}."
            )


ROW_SELECTIONS = ["manual", "auto"]


@dataclass
class SlineFromFrameConfig:
    pixel_offset_left: int
    pixel_offset_right: int
    # Rows summed into the spectral line. With row_selection = "manual" this
    # list is used as given. With "auto" the band is located automatically:
    # n_rows contiguous rows centred on the line's intensity centroid, chosen
    # ONCE and then frozen (see spectrum_fitting/row_selection.py — which rows
    # are summed shifts the fitted peaks by ~3-4 MHz per row, so the band must
    # not move between a scan's calibration and its samples).
    selected_rows: list[int]
    row_selection: str = "manual"
    n_rows: int = 13

    def __post_init__(self):
        if self.row_selection not in ROW_SELECTIONS:
            raise ValueError(
                f"Unknown row_selection '{self.row_selection}'. "
                f"Choose one of {ROW_SELECTIONS}."
            )

@dataclass
class FittingConfigs:
    sample_config: SampleFindPeaksConfig
    reference_config: FindPeaksConfig
    sline_config: SlineFromFrameConfig
    # Camera pixel-response constants. None = keep the fitter's current ones
    # (default so older callers that build a FittingConfigs keep working).
    psf_config: PsfConstants | None = None

FIND_PEAKS_TOML_PATH = Path(__file__).parent / "find_peaks_config.toml"

# Keys that used to live duplicated in the [sample]/[reference] sections
# (under their OLD pr_* names): the camera constants moved to the global
# [camera] section (now psf_*), na_* is sample-only, n_peaks was removed with
# the 4-peak mode. Dropped silently when an older TOML still carries them so
# those files keep loading; any other unknown key still raises.
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
    return SlineFromFrameConfig(**raw)

def save_config_section(path: Path, section: str, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)
    data[section] = asdict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)

# Global configuration instances. psf_config lives in ccd_characteristics
# (re-exported above) — the fitting TOML holds fit settings only.
find_peaks_sample_config = ThreadSafeConfig(load_config_section(FIND_PEAKS_TOML_PATH, "sample"))
find_peaks_reference_config = ThreadSafeConfig(load_config_section(FIND_PEAKS_TOML_PATH, "reference"))
sline_from_frame_config = ThreadSafeConfig(load_sline_from_frame_config(FIND_PEAKS_TOML_PATH))
