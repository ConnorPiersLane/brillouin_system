# config/find_peaks_config.py
from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig

FITTING_MODELS_SAMPLE = ["lorentzian", "lorentzian_quad_bg", "lorentzian_window", "voigt", "voigt_window", "asym_lorentzian", "asym_lorentzian_window", "na_lorentzian", "na_lorentzian_window", "na_gauss_lorentzian", "na_gauss_lorentzian_window"]
FITTING_MODELS_REFERENCE = ["lorentzian", "lorentzian_window", "voigt", "voigt_window", "asym_lorentzian", "asym_lorentzian_window"]

@dataclass
class FindPeaksConfig:
    prominence_fraction: float
    min_peak_width: int
    min_peak_height: int
    rel_height: float
    wlen_pixels: int
    fitting_model: str
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
