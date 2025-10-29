# init_coord_transformation.py
from pathlib import Path
import json
import numpy as np

from brillouin_system.eye_tracker.stereo_imaging.coord_transformer import SE3

# Directory structure same as your calibration
base = Path(__file__).resolve().parent
cfg = base / "stereo_configs"

# Path to your coordinate transform file
transform_path = cfg / "left_to_zaber.json"

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

try:
    d = _load_json(transform_path)
    R = np.asarray(d["R"], float).reshape(3, 3)
    t = np.asarray(d["t"], float).reshape(3)
    left_to_zaber = SE3(R, t)
    print(f"[CoordTransform] Loaded left_to_zaber (t = {t}, norm = {np.linalg.norm(t):.2f} mm)")
except Exception as e:
    print(f"[CoordTransform] Failed to load left_to_zaber.json: {e}")
    left_to_zaber = SE3(np.eye(3), np.zeros(3))  # fallback = identity

# Optional: make a registry like you do for stereo cameras
transforms = {"left_to_zaber": left_to_zaber}
