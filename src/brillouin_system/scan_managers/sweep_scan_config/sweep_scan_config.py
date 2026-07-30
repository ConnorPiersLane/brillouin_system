"""
Config for the in-out sweep scan (repeated find-measure-find cycles).

Same pattern as scan_managers/scanning_config: dataclass + TOML + a global
ThreadSafeConfig instance. Detection and motion parameters (speed, thresholds,
sample rate, min_samples_above, ...) are NOT duplicated here — the sweep scan
takes those from the shared axial ScanningConfig, so each crossing behaves
identically to the GUI reflection finder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli
import tomli_w

from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig


@dataclass
class SweepScanConfig:
    # ----------------------------
    # In-out sweep scan
    # ----------------------------
    n_repeats: int = 12               # number of in-out cycles
    approach_um: float = 300.0        # run-up distance past the plane on each side
    target_depth_um: float = 50.0     # measure at in-crossing + this (positive = inward)
    settle_s: float = 0.05            # lens settle time before the camera snap
    plausibility_gate_um: float = 750.0  # |crossing - previous plane| beyond this = discard


SWEEP_SCAN_TOML_PATH = Path(__file__).parent.resolve() / "sweep_scan_config.toml"


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(SweepScanConfig.__dataclass_fields__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _dataclass_to_toml_dict(cfg: SweepScanConfig) -> dict[str, Any]:
    return {
        "n_repeats": int(cfg.n_repeats),
        "approach_um": float(cfg.approach_um),
        "target_depth_um": float(cfg.target_depth_um),
        "settle_s": float(cfg.settle_s),
        "plausibility_gate_um": float(cfg.plausibility_gate_um),
    }


def load_sweep_scan_config(path: Path = SWEEP_SCAN_TOML_PATH, section: str = "sweep_scan") -> SweepScanConfig:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
        raw = data.get(section, {})
    except FileNotFoundError:
        raw = {}
    return SweepScanConfig(**_toml_to_kwargs(raw))


def save_sweep_scan_config(config: ThreadSafeConfig, path: Path = SWEEP_SCAN_TOML_PATH, section: str = "sweep_scan") -> None:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
    except FileNotFoundError:
        data = {}
    data[section] = _dataclass_to_toml_dict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)


sweep_scan_config = ThreadSafeConfig(load_sweep_scan_config())
