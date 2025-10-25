# stereo_init.py
from pathlib import Path

from brillouin_system.logging_utils.logging_setup import get_logger
from stereo_cameras import StereoCameras   # your renamed class
from calibration_models import Intrinsics, StereoExtrinsics, StereoCalibration
import numpy as np, json

log = get_logger(__name__)

# --- build once at import ---
base = Path(__file__).resolve().parent
cfg = base / "stereo_configs"

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

L = _load_json(cfg / "calibration_left.json")
R = _load_json(cfg / "calibration_right.json")
S = _load_json(cfg / "calibration_stereo.json")

intr_left = Intrinsics(
    K=np.array(L["camera"]["K"]),
    dist=np.array(L["camera"]["dist"]),
    image_size=tuple(L["image_size"]),
)
intr_right = Intrinsics(
    K=np.array(R["camera"]["K"]),
    dist=np.array(R["camera"]["dist"]),
    image_size=tuple(R["image_size"]),
)
extr = StereoExtrinsics(
    R=np.array(S["stereo"]["R"]),
    T=np.array(S["stereo"]["T"]),
    reference=S["stereo"].get("reference", "left"),
)

stereo_calib = StereoCalibration(left=intr_left, right=intr_right, extr=extr)
stereo_cameras = StereoCameras.from_stereo_calibration(stereo_calib)

baseline = float(np.linalg.norm(extr.T))
log.info(f"StereoCameras loaded (baseline = {baseline:.2f} mm, ref={extr.reference})")

