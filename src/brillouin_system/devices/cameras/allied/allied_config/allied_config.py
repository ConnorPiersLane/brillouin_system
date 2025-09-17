from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig

@dataclass
class AlliedConfig:
    id: str
    offset_x: int
    offset_y: int
    width: int
    height: int
    gain: float
    exposure: float
    gamma: float

# Path to config file
allied_config_toml_path = Path(__file__).parent.resolve() / "allied_config.toml"

def load_allied_settings(path: Path, section: str) -> AlliedConfig:
    """Load config for given section (e.g. 'left', 'right')."""
    with path.open("rb") as f:
        raw = tomli.load(f)[section]
    return AlliedConfig(**raw)

def save_allied_settings(path: Path, config_map: dict):
    """Save all configs from dict {section_name: ThreadSafeConfig}."""
    with path.open("rb") as f:
        data = tomli.load(f)

    for name, cfg in config_map.items():
        data[name] = asdict(cfg.get_raw())

    with path.open("wb") as f:
        tomli_w.dump(data, f)

# Global configs for left & right cameras
allied_config = {
    "DEV_000F315BC084": ThreadSafeConfig(load_allied_settings(allied_config_toml_path, "DEV_000F315BC084")),
    "DEV_000F315BC0A5": ThreadSafeConfig(load_allied_settings(allied_config_toml_path, "DEV_000F315BC0A5")),
}
