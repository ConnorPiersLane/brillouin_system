from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w

from brillouin_system.config.config import ThreadSafeConfig


@dataclass
class AndorConfig:
    advanced_gain_option: bool
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    vbin: int
    hbin: int
    pre_amp_mode: int
    vss_index: int
    temperature: float | str
    flip_image_horizontally: bool
    verbose: bool

# ---------- Load/save helpers ----------

andor_config_toml_path = Path(__file__).parent.resolve() / "andor_config.toml"



def load_andor_frame_settings(path: Path) -> AndorConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["andor_frame"]

    return AndorConfig(
        advanced_gain_option=raw["advanced_gain_option"],
        x_start=raw["x_start"],
        x_end=raw["x_end"],
        y_start=raw["y_start"],
        y_end=raw["y_end"],
        vbin=raw["vbin"],
        hbin=raw["hbin"],
        pre_amp_mode=raw["pre_amp_mode"],
        vss_index=raw["vss_index"],
        temperature=raw["temperature"],
        flip_image_horizontally=raw["flip_image_horizontally"],
        verbose=raw["verbose"],
    )



def save_andor_frame_settings(path: Path, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)

    data["andor_frame"] = asdict(config.get_raw())

    with path.open("wb") as f:
        tomli_w.dump(data, f)

# ---------- Global instances ----------

andor_frame_config = ThreadSafeConfig(load_andor_frame_settings(andor_config_toml_path))
