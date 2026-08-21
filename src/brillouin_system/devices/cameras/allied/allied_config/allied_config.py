from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.configs import CONFIG_DIR
from brillouin_system.helpers.thread_safe_config import LazyThreadSafeConfig, ThreadSafeConfig

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
allied_config_toml_path = CONFIG_DIR / "allied_config.toml"

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
    "left": LazyThreadSafeConfig(lambda: load_allied_settings(allied_config_toml_path, "left")),
    "right": LazyThreadSafeConfig(lambda: load_allied_settings(allied_config_toml_path, "right")),
}
