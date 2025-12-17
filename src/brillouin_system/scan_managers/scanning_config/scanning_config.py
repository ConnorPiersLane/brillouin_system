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
    exposure_time_for_reflection_finding: float = 0.05
    gain_for_reflection_finding: int = 1   # <-- ADD THIS
    reflection_threshold_value: float = 5000.0
    step_distance_um_for_reflection_finding: int = 20
    max_search_distance_um_for_reflection_finding: int = 2000
    step_after_finding_reflection_um: int = 20
    n_bg_images_for_reflection_finding: int = 10

    # ----------------------------
    # Scan Settings
    # ----------------------------
    max_scan_distance_um: int = 2000


# Path to the TOML config file
AXIAL_SCANNING_TOML_PATH = Path(__file__).parent.resolve() / "scanning_config.toml"


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Convert TOML section to AxialScanningConfig kwargs.

    Filters unknown keys so older TOML files (or copied templates) still load.
    """
    allowed = set(ScanningConfig.__dataclass_fields__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _dataclass_to_toml_dict(cfg: ScanningConfig) -> dict[str, Any]:
    """Convert an AxialScanningConfig instance to a simple dict for TOML."""
    return asdict(cfg)


def load_axial_scanning_config(
    path: Path, section: str = "axial_scanning"
) -> ScanningConfig:
    """
    Load AxialScanningConfig from a TOML file section.
    If the section is missing, fall back to dataclass defaults.
    """
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
        raw = data.get(section, {})
    except FileNotFoundError:
        raw = {}

    kwargs = _toml_to_kwargs(raw)
    return ScanningConfig(**kwargs)


def save_config_section(path: Path, section: str, config: ThreadSafeConfig) -> None:
    """
    Persist a ThreadSafeConfig[AxialScanningConfig] section back to TOML.
    """
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
    except FileNotFoundError:
        data = {}

    data[section] = _dataclass_to_toml_dict(config.get_raw())

    with path.open("wb") as f:
        tomli_w.dump(data, f)


# Single global configuration instance, loaded from TOML at import time
axial_scanning_config = ThreadSafeConfig(
    load_axial_scanning_config(AXIAL_SCANNING_TOML_PATH, "axial_scanning")
)
