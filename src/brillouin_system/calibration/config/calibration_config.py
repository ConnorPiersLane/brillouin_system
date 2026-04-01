from dataclasses import dataclass, asdict
from pathlib import Path
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

    # @property
    # def calibration_freqs(self) -> list[float]:
    #     # compute frequencies on the fly
    #     return [
    #         round(self.start + i * self.step, 6)
    #         for i in range(1000)
    #         if self.start + i * self.step <= self.stop
    #     ]
    @property
    def calibration_freqs(self) -> list[float]:
        """
        start = 4
        stop = 6
        step = 0.1
        gives [4.0, 5.0,
             4.1, 5.1,
             4.2, 5.2,
             ...
             4.9, 5.9,
             5.0]
        Returns:

        """
        freqs = []

        base = int(self.start)
        bands = list(range(base, int(self.stop)))  # e.g., [4, 5]

        i = 0
        while True:
            offset = i * self.step
            added_any = False

            for b in bands:
                val = b + offset
                if val < self.start:
                    continue
                if val <= self.stop:
                    freqs.append(round(val, 6))
                    added_any = True

            if not added_any:
                break

            i += 1

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
        ]
    }

    with path.open("wb") as f:
        tomli_w.dump(data, f)


# Global instance
calibration_config = ThreadSafeConfig(load_calibration_config(CALIBRATION_TOML_PATH))