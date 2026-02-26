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
    n_bg_samples: int = 10
    backstep_after_search_um: float = 0.0

    # refinement behavior
    do_refine: bool = False
    n_avg_samples: int = 20
    step_um: float = 5.0
    range_um: float = 50.0


AXIAL_SCANNING_TOML_PATH = Path(__file__).parent.resolve() / "scanning_config.toml"


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(ScanningConfig.__dataclass_fields__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _dataclass_to_toml_dict(cfg: ScanningConfig) -> dict[str, Any]:
    return {
        "n_sigma": cfg.n_sigma,
        "speed_um_s": cfg.speed_um_s,
        "max_search_distance_um": cfg.max_search_distance_um,
        "n_bg_samples": cfg.n_bg_samples,
        "backstep_after_search_um": cfg.backstep_after_search_um,
        "do_refine": bool(cfg.do_refine),
        "n_avg_samples": cfg.n_avg_samples,
        "step_um": cfg.step_um,
        "range_um": cfg.range_um,
    }


def load_axial_scanning_config(
    path: Path,
    section: str = "axial_scanning",
) -> ScanningConfig:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
        raw = data.get(section, {})
    except FileNotFoundError:
        raw = {}

    return ScanningConfig(**_toml_to_kwargs(raw))


def save_config_section(
    path: Path,
    section: str,
    config: ThreadSafeConfig,
) -> None:
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