# config/find_peaks_config.py
from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig

# The model name selects the LINESHAPE only. Windowing (use_window) and the
# baseline (background) are independent toggles that apply to any lineshape.
FITTING_MODELS_SAMPLE = [
    "lorentzian",
    "na_lorentzian",
    "na_gauss_lorentzian",
    "pixel_response",
    "prm0",
    "prm1",
]
# prm0/prm1 are offered here too: a scan's calibration is re-fitted with this
# config (calibration_for_scan), so picking the preset gives the sidebands the
# same per-peak baseline, window and beta = 3.0 as the samples. That matters for
# the width recipe — the instrument width subtracted from a sample linewidth
# should be fitted the way the sample was. Plain "pixel_response" is the lighter
# choice when only the centre convention has to match, which is all the
# model-mixing guard requires.
FITTING_MODELS_REFERENCE = [
    "lorentzian",
    "voigt",
    "pixel_response",
    "prm0",
    "prm1",
]

# Preset names for the validated production recipes (2026-08-03 decision).
# A preset pins lineshape + background + window + beta together, because the
# combination is what was validated — the width recipe is only valid at
# beta = 3.0 (the halo appears in the wings from ~3.5 gamma out). Use the
# long-form (fitting_model + background) to deviate deliberately.
#   prm0  pixel_response + per-peak flat offset   — unbiased WIDTHS; single-
#         peak centers carry the halo's odd moment (distance is unaffected).
#   prm1  pixel_response + per-peak linear bg     — unbiased CENTERS (the
#         slope absorbs the halo's odd moment) at the cost of a state-
#         dependent width-gap fabrication (up to ~+2.5 MHz, gold session).
# Production usage: fit with prm1; the prm0 companion width-gap difference is
# the QA indicator (accept widths when |gap_prm1 - gap_prm0| <= ~1.5 MHz).
MODEL_PRESETS = {
    "prm0": {"fitting_model": "pixel_response", "background": "flat_per_peak",
             "use_window": True, "beta": 3.0},
    "prm1": {"fitting_model": "pixel_response", "background": "linear_per_peak",
             "use_window": True, "beta": 3.0},
}

# Baseline under the peaks:
#   flat            one shared constant offset (the long-standing behaviour)
#   linear          one shared constant + slope across the whole fit domain
#   flat_per_peak   each peak gets its own constant offset over its own
#                   window — the width-safe per-peak baseline (a free slope
#                   is odd-symmetric: it corrects the CENTER exactly but
#                   leaks into the WIDTH via covariance, fabricating up to
#                   ~+2.5 MHz of L-R width gap; closure-tested 2026-08-02).
#   linear_per_peak each peak gets its own constant + slope over its own
#                   window — this is the one that removes the L-R split in
#                   water (measured 2026-07: -3.3 -> -0.4 MHz), because the
#                   bias comes from the LOCAL gradient under each peak.
# Costs ~10% in single-frame distance precision. Validated on liquids only:
# cornea splits are already unbiased and get WORSE with a background term.
BACKGROUNDS = ["flat", "linear", "flat_per_peak", "linear_per_peak"]

# Models that were removed, with the migration hint shown if one is still set.
_REMOVED_MODELS = {
    "asym_lorentzian": "tested on calibration data and rejected (it triples the "
                       "residual sine); use 'pixel_response' for an asymmetric "
                       "instrument lineshape",
    "lorentzian_quad_bg": "never implemented; use background='linear_per_peak' "
                          "(a quadratic baseline was tested and is degenerate "
                          "with the Lorentzian wings)",
}

# Legacy model names that folded a baseline choice into the lineshape name.
_LEGACY_BACKGROUND_MODELS = {
    "lorentzian_linear_bg": ("lorentzian", "linear_per_peak"),
}


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
    # NA-integrated models only (0.0 = unset -> those models refuse to run).
    # na_collection: hard aperture clip as an NA (alpha = arcsin(NA/n)).
    #   - na_lorentzian* (uniform pupil): the EFFECTIVE NA, calibrated per
    #     session on water (absorbs the coupling apodization).
    #   - na_gauss_lorentzian*: the NOMINAL objective NA (physical pupil edge);
    #     the apodization is modeled explicitly via the two fields below.
    # na_gauss_lorentzian* only — Gaussian fiber-coupling weight
    # exp(-2 (v/v0)^2), v0 = arcsin(sin(arctan((D/2)/f))/n):
    #   na_beam_diameter_mm: D, 1/e^2 diameter of the collection-fiber mode at
    #     the objective pupil (collimator output beam; F810APC-780 nominal
    #     7.5 mm). The session-calibration knob: tune on water (effective < nominal).
    #   na_focal_length_mm: f, focal length of the OBJECTIVE (20X: 10, 5X: 40).
    # na_n_sample: refractive index of the sample medium.
    na_collection: float = 0.0
    na_beam_diameter_mm: float = 0.0
    na_focal_length_mm: float = 0.0
    na_n_sample: float = 1.33
    # 'pixel_response' model only (reference/calibration peaks). Frozen camera
    # pixel-response constants, NOT fitted per frame:
    #   pr_sigma_px    Gaussian charge-diffusion blur.
    #   pr_tau_*_px    one-sided exponential readout smear, per peak, toward
    #                  higher pixel numbers (the charge-transfer direction).
    # Measured 2026-07 on the fine EOM sweeps: 0.25 / 0.40 / 0.20 px, stable
    # across 6 calibrations over 7 weeks. Re-measure after any camera/ROI
    # change; the model refuses to run while all three are 0.
    pr_sigma_px: float = 0.0
    pr_tau_left_px: float = 0.0
    pr_tau_right_px: float = 0.0

    def __post_init__(self):
        model = str(self.fitting_model)
        # Legacy '<model>_window' names -> base name + use_window.
        if model.endswith("_window"):
            model = model[: -len("_window")]
            self.use_window = True
        # Legacy names that folded the baseline into the lineshape name.
        if model in _LEGACY_BACKGROUND_MODELS:
            model, self.background = _LEGACY_BACKGROUND_MODELS[model]
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
    sample_config: FindPeaksConfig
    reference_config: FindPeaksConfig
    sline_config: SlineFromFrameConfig

FIND_PEAKS_TOML_PATH = Path(__file__).parent / "find_peaks_config.toml"

def load_config_section(path: Path, section: str) -> FindPeaksConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)[section]
    return FindPeaksConfig(**raw)

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

# Global configuration instances
find_peaks_sample_config = ThreadSafeConfig(load_config_section(FIND_PEAKS_TOML_PATH, "sample"))
find_peaks_reference_config = ThreadSafeConfig(load_config_section(FIND_PEAKS_TOML_PATH, "reference"))
sline_from_frame_config = ThreadSafeConfig(load_sline_from_frame_config(FIND_PEAKS_TOML_PATH))
