
from pathlib import Path
import json
import numpy as np

from brillouin_system.eye_tracker.stereo_imaging.se3 import SE3

# Directory structure same as your calibration
base = Path(__file__).resolve().parent
cfg = base / "stereo_configs"

# Path to your coordinate transform file
transform_path = cfg / "left_to_zaber.json"

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


d = _load_json(transform_path)
R = np.asarray(d["R"], float).reshape(3, 3)
t = np.asarray(d["t"], float).reshape(3)
left_to_ref = SE3(R, t)
print(f"[CoordTransform] Loaded left_to_zaber (t = {t}, norm = {np.linalg.norm(t):.2f} mm)")



