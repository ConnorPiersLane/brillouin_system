# config/pupil_fit_config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import tomli
import tomli_w
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig  # same helper you use elsewhere

@dataclass
class PupilFitConfig:
    """
    Minimal, robust knobs for fast pupil ellipse fitting.
    Keep it small so you rarely need to touch it.
    """
    # Preprocessing
    gaussian_ksize: int = 5             # must be odd; 5 is a good default
    # Threshold: pupil is darker than iris/sclera -> invert + Otsu
    use_otsu: bool = True
    # Morphology
    close_kernel: int = 3               # 3x3 elliptical kernel
    close_iterations: int = 1
    # Contour gating
    min_area_frac: float = 5e-4         # reject too small blobs (fraction of processed image area)
    max_area_frac: float = 0.5          # reject too large blobs
    max_bbox_aspect: float = 3.5        # reject extreme skinny blobs
    # Downscaling for speed (rescaled back to original coords)
    scale: float = 0.5                  # 0.4–0.6 is typically great
    # Optional ROI (x,y,w,h). Use when you can crop to the eye region for speed & robustness.
    roi: Optional[Tuple[int, int, int, int]] = None

PUPIL_FIT_TOML_PATH = Path(__file__).parent / "pupil_fit_config.toml"

def _toml_to_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert TOML values into constructor kwargs (e.g., ROI list -> tuple)."""
    out = dict(raw)
    if "roi" in out and out["roi"] is not None:
        roi = out["roi"]
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            out["roi"] = tuple(int(v) for v in roi)
        else:
            out["roi"] = None
    return out

def _dataclass_to_toml_dict(cfg: PupilFitConfig) -> Dict[str, Any]:
    """Convert dataclass to TOML-friendly dict (tuples -> lists)."""
    d = asdict(cfg)
    if d.get("roi") is not None:
        d["roi"] = list(d["roi"])
    return d

def load_pupil_fit_config(path: Path, section: str) -> PupilFitConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)[section]
    return PupilFitConfig(**_toml_to_kwargs(raw))

def save_config_section(path: Path, section: str, config: ThreadSafeConfig):
    """Persist a ThreadSafeConfig[PupilFitConfig] section back to TOML (like your find_peaks saver)."""
    with path.open("rb") as f:
        data = tomli.load(f)
    data[section] = _dataclass_to_toml_dict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)

# Global configuration instances (two cameras typical: left/right)
left_eye_pupil_fit_config  = ThreadSafeConfig(load_pupil_fit_config(PUPIL_FIT_TOML_PATH, "left_eye"))
right_eye_pupil_fit_config = ThreadSafeConfig(load_pupil_fit_config(PUPIL_FIT_TOML_PATH, "right_eye"))
