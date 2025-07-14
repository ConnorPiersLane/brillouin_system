# calibration_config.py
from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
from brillouin_system.config.config import ThreadSafeConfig

CALIBRATION_TOML_PATH = Path(__file__).parent / "calibration_config.toml"

@dataclass
class CalibrationConfig:
    n_per_freq: int
    degree: int
    start: float
    stop: float
    step: float
    reference: str  # "left", "right", or "distance"

    @property
    def calibration_freqs(self) -> list[float]:
        # compute frequencies on the fly
        return [round(self.start + i * self.step, 6) for i in range(1000)
                if self.start + i * self.step <= self.stop]

def load_calibration_config(path: Path = CALIBRATION_TOML_PATH) -> CalibrationConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["calibration"]

    return CalibrationConfig(
        n_per_freq=raw["n_per_freq"],
        degree=raw["degree"],
        start=raw["start"],
        stop=raw["stop"],
        step=raw["step"],
        reference=raw["reference"]
    )

def save_calibration_config(path: Path, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)

    # Don't save the computed list, only the inputs
    raw = asdict(config.get_raw())
    data["calibration"] = {k: raw[k] for k in ["n_per_freq", "degree", "start", "stop", "step", "reference"]}

    with path.open("wb") as f:
        tomli_w.dump(data, f)

# Global instance
calibration_config = ThreadSafeConfig(load_calibration_config(CALIBRATION_TOML_PATH))

