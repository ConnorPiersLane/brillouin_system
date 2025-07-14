# config/find_peaks_config.py
from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.config.thread_safe_config import ThreadSafeConfig

FITTING_MODELS_SAMPLE = ["lorentzian", "gaussian", "voigt"]
FITTING_MODELS_REFERENCE = ["lorentzian", "gaussian"]

@dataclass
class FindPeaksConfig:
    prominence_fraction: float
    min_peak_width: int
    min_peak_height: int
    rel_height: float
    wlen_pixels: int
    fitting_model: str

FIND_PEAKS_TOML_PATH = Path(__file__).parent / "find_peaks_config.toml"

def load_config_section(path: Path, section: str) -> FindPeaksConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)[section]
    return FindPeaksConfig(**raw)

def save_config_section(path: Path, section: str, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)
    data[section] = asdict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)



# Load the actual config data once at init
find_peaks_sample_config = ThreadSafeConfig(load_config_section(FIND_PEAKS_TOML_PATH, "sample"))
find_peaks_reference_config = ThreadSafeConfig(load_config_section(FIND_PEAKS_TOML_PATH, "reference"))

