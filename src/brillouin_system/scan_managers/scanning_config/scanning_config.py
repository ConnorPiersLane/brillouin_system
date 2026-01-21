from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import tomli
import tomli_w

from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig


@dataclass
class ScanningConfig:
    # ----------------------------
    # Find Reflection Settings
    # ----------------------------
    exposure: float = 0.05
    gain: int = 1
    n_sigma: int = 6
    speed_um_s: float = 1000
    max_search_distance_um: float = 2000
    n_bg_images: int = 10

    # ----------------------------
    # Scan Settings
    # ----------------------------
    max_scan_distance_um: int = 2000


AXIAL_SCANNING_TOML_PATH = Path(__file__).parent.resolve() / "scanning_config.toml"


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(ScanningConfig.__dataclass_fields__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _dataclass_to_toml_dict(cfg: ScanningConfig) -> dict[str, Any]:
    return asdict(cfg)


def load_axial_scanning_config(
    path: Path, section: str = "axial_scanning"
) -> ScanningConfig:
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

    data[section] = _dataclass_to_toml_dict(config.get_raw())

    with path.open("wb") as f:
        tomli_w.dump(data, f)


axial_scanning_config = ThreadSafeConfig(
    load_axial_scanning_config(AXIAL_SCANNING_TOML_PATH, "axial_scanning")
)
