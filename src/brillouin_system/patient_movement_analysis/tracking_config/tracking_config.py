"""
Config for the cornea sweep tracker (patient movement analysis).

Same pattern as scan_managers/scanning_config: dataclass + TOML + a global
ThreadSafeConfig instance. The reflection-detection parameters (thresholds,
sample rate, debounce, min_samples_above, ...) are NOT duplicated here — the
tracker takes those from the shared axial ScanningConfig, so detection behaves
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
class TrackingConfig:
    # ----------------------------
    # Cornea sweep tracker
    # ----------------------------
    sweep_amplitude_um: float = 100.0   # sweep +- this around the surface
    sweep_speed_um_s: float = 2000.0    # sweep speed (use the characterized speed)
    bg_acqui_s: float = 0.3             # background acquisition before tracking starts
    recenter: bool = True               # re-center sweep window on the tracked surface
    recenter_avg_points: int = 5        # moving average length for re-centering
    max_track_time_s: float = 300.0     # safety cap on one tracking session


TRACKING_TOML_PATH = Path(__file__).parent.resolve() / "tracking_config.toml"


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(TrackingConfig.__dataclass_fields__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _dataclass_to_toml_dict(cfg: TrackingConfig) -> dict[str, Any]:
    return {
        "sweep_amplitude_um": float(cfg.sweep_amplitude_um),
        "sweep_speed_um_s": float(cfg.sweep_speed_um_s),
        "bg_acqui_s": float(cfg.bg_acqui_s),
        "recenter": bool(cfg.recenter),
        "recenter_avg_points": int(cfg.recenter_avg_points),
        "max_track_time_s": float(cfg.max_track_time_s),
    }


def load_tracking_config(path: Path = TRACKING_TOML_PATH, section: str = "tracking") -> TrackingConfig:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
        raw = data.get(section, {})
    except FileNotFoundError:
        raw = {}
    return TrackingConfig(**_toml_to_kwargs(raw))


def save_tracking_config(config: ThreadSafeConfig, path: Path = TRACKING_TOML_PATH, section: str = "tracking") -> None:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
    except FileNotFoundError:
        data = {}
    data[section] = _dataclass_to_toml_dict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)


tracking_config = ThreadSafeConfig(load_tracking_config())
