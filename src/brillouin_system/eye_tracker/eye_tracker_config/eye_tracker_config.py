# config/eye_tracker_config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal
import tomli
import tomli_w
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig


@dataclass
class EyeTrackerConfig:
    """
    Minimal, robust knobs for fast pupil ellipse fitting.
    Keep it small so you rarely need to touch it.

    ROI notes:
    - roi_*_center_xy are (x, y) centers in pixels in the full image.
    - roi_*_width_height are (width, height) in pixels.
    """
    sleep: bool = False

    # Per-eye thresholds
    binary_threshold_left: int = 20
    binary_threshold_right: int = 20

    # Per-eye ROIs
    roi_left_center_xy: tuple[int, int] = (500, 500)
    roi_left_width_height: tuple[int, int] = (100, 100)
    roi_right_center_xy: tuple[int, int] = (500, 500)
    roi_right_width_height: tuple[int, int] = (100, 100)
    apply_roi: bool = True

    # Saving controls
    save_images_path: str = ""
    max_saving_freq_hz: int = 5
    save_images: bool = False

    # Ellipse fitting & overlay
    do_ellipse_fitting: bool = False
    overlay_ellipse: bool = False

    # Which image is returned by the pipeline
    frame_returned: Literal["original", "binary", "floodfilled", "contour"] = "original"


PUPIL_FIT_TOML_PATH = Path(__file__).parent / "eye_tracker_config.toml"


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert TOML section to EyeTrackerConfig kwargs (no special coercions needed)."""
    # Note: TOML arrays will come in as lists; EyeTrackerConfig will happily accept them
    # for tuple-annotated fields, so we don't need to coerce explicitly.
    return dict(raw)


def _dataclass_to_toml_dict(cfg: EyeTrackerConfig) -> dict[str, Any]:
    """Convert dataclass to TOML-friendly dict (direct asdict)."""
    return asdict(cfg)


def load_eye_tracker_config(path: Path, section: str = "eye_tracker") -> EyeTrackerConfig:
    with path.open("rb") as f:
        data = tomli.load(f)
    raw = data.get(section, {})
    return EyeTrackerConfig(**_toml_to_kwargs(raw))


def save_config_section(path: Path, section: str, config: ThreadSafeConfig):
    """Persist a ThreadSafeConfig[EyeTrackerConfig] section back to TOML."""
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
    except FileNotFoundError:
        data = {}

    data[section] = _dataclass_to_toml_dict(config.get_raw())

    with path.open("wb") as f:
        tomli_w.dump(data, f)


# Single global configuration instance
eye_tracker_config = ThreadSafeConfig(
    load_eye_tracker_config(PUPIL_FIT_TOML_PATH, "eye_tracker")
)
