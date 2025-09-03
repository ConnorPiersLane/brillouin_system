from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig

@dataclass
class LEDConfig:
    intensity_led_white_below: int
    intensity_led_blue_385_below: int
    intensity_led_red_625_below: int
    intensity_led_white_top: int
    is_led_white_below: bool
    is_led_blue_385_below: bool
    is_led_red_625_below: bool
    is_led_white_top: bool

led_config_toml_path = Path(__file__).parent.resolve() / "led_config.toml"

def load_led_settings(path: Path) -> LEDConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["led"]
    return LEDConfig(**raw)

def save_led_settings(path: Path, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)
    data["led"] = asdict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)

led_config = ThreadSafeConfig(load_led_settings(led_config_toml_path))
