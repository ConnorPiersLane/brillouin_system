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
]
FITTING_MODELS_REFERENCE = [
    "lorentzian",
    "voigt",
    "pixel_response",
]

# Baseline under the peaks:
#   flat            one shared constant offset (the long-standing behaviour)
#   linear          one shared constant + slope across the whole fit domain
#   linear_per_peak each peak gets its own constant + slope over its own
#                   window — this is the one that removes the L-R split in
#                   water (measured 2026-07: -3.3 -> -0.4 MHz), because the
#                   bias comes from the LOCAL gradient under each peak.
# Costs ~10% in single-frame distance precision. Validated on liquids only:
# cornea splits are already unbiased and get WORSE with a background term.
BACKGROUNDS = ["flat", "linear", "linear_per_peak"]

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
class SlineFromFrameConfig:
    pixel_offset_left: int
    pixel_offset_right: int
    selected_rows: list[int]

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
