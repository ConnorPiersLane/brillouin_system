"""
Single-camera calibration utilities (pinhole & fisheye).

- Calibrate from a list of images that independently contain the checkerboard.
- Returns CameraResult and image_size.
- Save parameters to TOML (one file per camera).

Designed to interoperate with calibrate_stereo.py and the GUI.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

try:
    import tomllib as _tomllib  # py3.11+
except Exception:
    _tomllib = None
try:
    import tomli_w as _tomli_w
except Exception:
    _tomli_w = None

# ---------------- TOML writer ----------------
def _write_toml(data: dict, path: str) -> None:
    if _tomli_w is not None:
        with open(path, "wb") as f:
            f.write(_tomli_w.dumps(data).encode("utf-8"))
        return
    def _fmt(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return repr(float(v))
        if isinstance(v, str):
            return '"' + v.replace('"', '\\"') + '"'
        if isinstance(v, (list, tuple, np.ndarray)):
            return "[" + ", ".join(_fmt(x) for x in np.asarray(v).tolist()) + "]"
        return '"' + str(v) + '"'
    lines = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"[{k}]")
            for k2, v2 in v.items():
                lines.append(f"{k2} = {_fmt(v2)}")
            lines.append("")
        else:
            lines.append(f"{k} = {_fmt(v)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------------- Data classes ----------------
@dataclass
class MonoCalibConfig:
    model: str = "pinhole"
    cols: int = 9
    rows: int = 6
    square_size_mm: float = 25.0

@dataclass
class CameraResult:
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    rms: float

# ---------------- Core functions ----------------
def prepare_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)
    return objp

def detect_corners(img: np.ndarray, pattern_size: Tuple[int, int]) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    )
    return corners

def calibrate_single(images: List[np.ndarray], config: MonoCalibConfig) -> Tuple[CameraResult, Tuple[int,int]]:
    pattern_size = (int(config.cols), int(config.rows))
    objp = prepare_object_points(config.cols, config.rows, config.square_size_mm)
    objpoints, imgpoints = [], []
    image_size = None
    for im in images:
        c = detect_corners(im, pattern_size)
        if c is None:
            continue
        if image_size is None:
            h, w = im.shape[:2]
            image_size = (w, h)
        objpoints.append(objp.astype(np.float32))
        imgpoints.append(c.astype(np.float32))

    if not objpoints:
        raise RuntimeError("No valid checkerboard detections for mono calibration.")

    if config.model.lower() == "pinhole":
        # Stage 1: conservative (prevents overfitting)
        flags_cons = (cv2.CALIB_ZERO_TANGENT_DIST |
                      cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None, flags=flags_cons
        )

        # Stage 2 (tiny relaxation): free K3 only, accept only if it really helps
        flags_k3 = (cv2.CALIB_ZERO_TANGENT_DIST |
                    cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

        ret2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, K.copy(), dist.copy(), flags=flags_k3
        )

        # Gate: require a real RMS improvement AND principal point moving toward center
        cx0, cy0 = K[0, 2], K[1, 2]
        cx1, cy1 = K2[0, 2], K2[1, 2]
        cx_target, cy_target = image_size[0] / 2.0, image_size[1] / 2.0

        def closer(a0, a1, tgt):  # did we move closer to the target?
            return abs(a1 - tgt) <= abs(a0 - tgt) * 1.05  # allow tiny noise (5%)

        improved_rms = (ret2 < ret * 0.9)  # ~10% better
        moved_toward_center = closer(cx0, cx1, cx_target) and closer(cy0, cy1, cy_target)
        k3_reasonable = abs(float(dist2.ravel()[2])) < 1.5 if dist2.size >= 3 else True  # sanity bound

        if improved_rms and moved_toward_center and k3_reasonable:
            K, dist, rvecs, tvecs, ret = K2, dist2, rvecs2, tvecs2, ret2

        R, _ = cv2.Rodrigues(rvecs[-1]); t = tvecs[-1].reshape(3)
        return CameraResult(K, dist.reshape(-1), R, t, float(ret)), image_size

    # Fisheye unchanged
    objp_fe = [op.reshape(-1,1,3).astype(np.float64) for op in objpoints]
    imgp_fe = [ip.astype(np.float64) for ip in imgpoints]
    K = np.zeros((3,3)); D = np.zeros((4,1))
    fe_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                cv2.fisheye.CALIB_CHECK_COND |
                cv2.fisheye.CALIB_FIX_SKEW)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objp_fe, imgp_fe, image_size, K, D, None, None, flags=fe_flags)
    R, _ = cv2.Rodrigues(rvecs[-1]); t = tvecs[-1].reshape(3)
    return CameraResult(K, D.reshape(-1), R, t, float(rms)), image_size

def save_camera_json(
    out_path: str,
    res: CameraResult,
    image_size: Tuple[int,int],
    config: MonoCalibConfig,
):
    data = {
        "config": {
            "model": config.model,
            "cols": int(config.cols),
            "rows": int(config.rows),
            "square_size_mm": float(config.square_size_mm),
        },
        "image_size": [int(image_size[0]), int(image_size[1])],
        "camera": {
            "K": np.asarray(res.K, dtype=float).tolist(),
            "dist": np.asarray(res.dist, dtype=float).tolist(),
            "R": np.asarray(res.R, dtype=float).tolist(),
            "t": np.asarray(res.t, dtype=float).tolist(),
            "rms": float(res.rms),
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
