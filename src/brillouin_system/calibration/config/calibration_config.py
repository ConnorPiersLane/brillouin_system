from dataclasses import dataclass, asdict
from pathlib import Path
import random

import tomli
import tomli_w
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig

CALIBRATION_TOML_PATH = Path(__file__).parent / "calibration_config.toml"


@dataclass
class CalibrationConfig:
    n_per_freq: int
    degree: int
    start: float
    stop: float
    step: float
    reference: str  # "left", "right", or "distance"
    mode: str  # "poly" or "interp"
    # How the reference peak centers are obtained during calibrate():
    # "lorentzian" = classic lorentzian_window fits (default, old behavior)
    # "psf"        = two-stage: bootstrap lorentzian fits, reconstruct the
    #                empirical per-order PSF from the sideband sweep, then
    #                re-fit all centers with a shifted-PSF model. Also stores
    #                the PSFs for the DHO sample fit.
    centering: str = "lorentzian"

    @property
    def calibration_freqs(self) -> list[float]:
        n_steps = int(round((self.stop - self.start) / self.step))

        freqs = [
            round(self.start + i * self.step, 6)
            for i in range(n_steps + 1)
            if self.start + i * self.step <= self.stop + 1e-9
        ]

        random.shuffle(freqs)
        return freqs



def load_calibration_config(path: Path = CALIBRATION_TOML_PATH) -> CalibrationConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["calibration"]

    return CalibrationConfig(
        n_per_freq=raw["n_per_freq"],
        degree=raw["degree"],
        start=raw["start"],
        stop=raw["stop"],
        step=raw["step"],
        reference=raw["reference"],
        mode=raw["mode"],
        centering=raw.get("centering", "lorentzian"),
    )


def save_calibration_config(path: Path, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)

    raw = asdict(config.get_raw())
    data["calibration"] = {
        k: raw[k]
        for k in [
            "n_per_freq",
            "degree",
            "start",
            "stop",
            "step",
            "reference",
            "mode",
            "centering",
        ]
    }

    with path.open("wb") as f:
        tomli_w.dump(data, f)


# Global instance
calibration_config = ThreadSafeConfig(load_calibration_config(CALIBRATION_TOML_PATH))