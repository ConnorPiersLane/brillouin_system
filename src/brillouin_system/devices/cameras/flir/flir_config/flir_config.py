from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.config.config import ThreadSafeConfig

@dataclass
class FLIRConfig:
    offset_x: int
    offset_y: int
    width: int
    height: int
    pixel_format: str
    gain: int
    exposure: int
    gamma: float

# Camera Values for the:
# Blackfly S BFS-U3-70S7M
_max_gain = 47.99
_min_gain = 0.0
_max_exposure_time = 29999998.0
_min_exposure_time = 9
_max_gamma = 4.0
_min_gamma = 0.25

# Path to config file
flir_config_toml_path = Path(__file__).parent.resolve() / "flir_config.toml"

def load_flir_settings(path: Path) -> FLIRConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["flir"]
    return FLIRConfig(**raw)

def save_flir_settings(path: Path, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)
    data["flir"] = asdict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)

# Global config instance
flir_config = ThreadSafeConfig(load_flir_settings(flir_config_toml_path))
