from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli
import tomli_w

from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig


@dataclass
class ScanningConfig:
    # ----------------------------
    # Axial Scanning
    # ----------------------------
    n_sigma: int = 10
    speed_um_s: float = 5000.0
    max_search_distance_um: float = 5000.0
    background_acquisition_time_ms: int = 10
    backstep_after_search_um: float = 0.0

    # refinement behavior
    do_refine: bool = False
    point_acquisition_time_ms: int = 20
    step_um: float = 5.0
    range_um: float = 50.0

    # analysis behavior
    n_max_values: int = 5  # mean of top-N values; if fewer exist, mean of all


AXIAL_SCANNING_TOML_PATH = Path(__file__).parent.resolve() / "scanning_config.toml"


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a TOML section dict into ScanningConfig kwargs.

    - Filters unknown keys.
    - Includes small backward-compat shims for older key names.
    """
    allowed = set(ScanningConfig.__dataclass_fields__.keys())

    # Backward compatibility (older config files)
    if "background_acquisition_time_ms" not in raw and "n_bg_samples" in raw:
        raw = dict(raw)
        raw["background_acquisition_time_ms"] = raw["n_bg_samples"]

    if "point_acquisition_time_ms" not in raw and "n_avg_samples" in raw:
        raw = dict(raw)
        raw["point_acquisition_time_ms"] = raw["n_avg_samples"]

    # If n_max_values missing, dataclass default (5) will apply automatically.

    return {k: v for k, v in raw.items() if k in allowed}


def _dataclass_to_toml_dict(cfg: ScanningConfig) -> dict[str, Any]:
    return {
        "n_sigma": cfg.n_sigma,
        "speed_um_s": cfg.speed_um_s,
        "max_search_distance_um": cfg.max_search_distance_um,
        "background_acquisition_time_ms": cfg.background_acquisition_time_ms,
        "backstep_after_search_um": cfg.backstep_after_search_um,
        "do_refine": bool(cfg.do_refine),
        "point_acquisition_time_ms": cfg.point_acquisition_time_ms,
        "step_um": cfg.step_um,
        "range_um": cfg.range_um,
        "n_max_values": int(cfg.n_max_values),
    }


def load_axial_scanning_config(path: Path, section: str = "axial_scanning") -> ScanningConfig:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
        raw = data.get(section, {})
    except FileNotFoundError:
        raw = {}

    return ScanningConfig(**_toml_to_kwargs(raw))


def save_config_section(path: Path, section: str, config: ThreadSafeConfig) -> None:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
    except FileNotFoundError:
        data = {}

    cfg: ScanningConfig = config.get_raw()
    data[section] = _dataclass_to_toml_dict(cfg)

    with path.open("wb") as f:
        tomli_w.dump(data, f)


axial_scanning_config = ThreadSafeConfig(
    load_axial_scanning_config(AXIAL_SCANNING_TOML_PATH, "axial_scanning")
)