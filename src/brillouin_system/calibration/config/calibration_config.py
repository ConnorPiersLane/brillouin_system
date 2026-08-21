from dataclasses import dataclass, asdict
from math import gcd
from pathlib import Path
from random import Random

import tomli
import tomli_w
from brillouin_system.configs import CONFIG_DIR
from brillouin_system.helpers.thread_safe_config import LazyThreadSafeConfig, ThreadSafeConfig

CALIBRATION_TOML_PATH = CONFIG_DIR / "calibration_config.toml"


def _strided_order(freqs: list[float]) -> list[float]:
    """Scrambled but DETERMINISTIC measurement order: visit every k-th
    frequency of the sorted grid, with k coprime to n so each one is hit
    exactly once (k ~ 0.38*n, the golden-ratio stride, keeps consecutive
    measurements far apart on the axis).

    Scrambled, so a slow drift (temperature, alignment, laser power) cannot
    masquerade as a frequency-dependent effect the way a monotonic sweep
    would let it. Deterministic — a pure function of the grid — so repeated
    reads give the same order and a scan's sequence is reproducible.

    Works for any n: the coprime search always terminates (n-1 is coprime
    to n), and a coprime stride visits every index exactly once. A few
    small grids (n = 3, 4, 6) have NO well-spread stride — their only
    coprimes are +-1 mod n, i.e. a monotone sweep — so those fall back to
    a seeded shuffle, which is equally deterministic.
    """
    n = len(freqs)
    if n <= 2:
        return list(freqs)
    k = max(2, round(0.382 * n))
    while gcd(k, n) != 1:
        k += 1
    if min(k, n - k) <= 1:
        # Stride is +-1 mod n = a monotone sweep; scramble deterministically.
        order = list(freqs)
        Random(n).shuffle(order)
        return order
    return [freqs[(i * k) % n] for i in range(n)]


@dataclass
class CalibrationConfig:
    n_per_freq: int
    degree: int
    start: float
    stop: float
    step: float
    # Which observable the live display / analyzer reports: "left", "right",
    # "distance" (inner pair — the absolute anchor), or "combined" (the
    # inverse-variance combination of all four orders; needs n_peaks = 4 in
    # BOTH fitting sections, shows N/A otherwise).
    reference: str
    # Whether to store the raw calibration reference frames on each axial
    # scan (AxialScan.calibration_data). Adds frames to every saved scan, but
    # re-fitting a scan against its OWN calibration (the production
    # re-analysis path) is only possible when they are stored.
    save_calibration_frames: bool = True

    @property
    def calibration_freqs(self) -> list[float]:
        """The sweep grid in its measurement order (see _strided_order)."""
        n_steps = int(round((self.stop - self.start) / self.step))

        freqs = [
            round(self.start + i * self.step, 6)
            for i in range(n_steps + 1)
            if self.start + i * self.step <= self.stop + 1e-9
        ]

        return _strided_order(freqs)



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
        save_calibration_frames=raw.get("save_calibration_frames", True),
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
            "save_calibration_frames",
        ]
    }

    with path.open("wb") as f:
        tomli_w.dump(data, f)


# Global instance
calibration_config = LazyThreadSafeConfig(lambda: load_calibration_config(CALIBRATION_TOML_PATH))