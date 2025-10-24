# calibrate_stereo.py
"""
Stereo extrinsics-only calibration utilities.

Usage pattern (high level):
  1) Run mono calibrations elsewhere -> produce two CameraResult objects (left_res, right_res).
  2) Build/scan pairs and filter valid ones with the helpers here.
  3) Call stereo_calibrate_from_pairs(valid_pairs, config, left_res, right_res).
  4) save_stereo_json(...)  # writes ONLY stereo (R,T,E,F) + minimal metadata.

This file intentionally does NOT recompute mono intrinsics. It fixes them and
solves only for R,T (and E,F) via cv2.stereoCalibrate with CALIB_FIX_INTRINSIC.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2

# Reuse your mono dataclasses + helpers
# (These names/types match your existing files.)
from brillouin_system.eye_tracker.stereo_calibration.calibrate_single import (
    MonoCalibConfig,             # only for reference in GUI, not used here
    CameraResult,                # K, dist, R, t, rms
    detect_corners,              # chessboard detection
    prepare_object_points,       # (cols,rows,size)->object points grid
)  # :contentReference[oaicite:2]{index=2}


# -----------------------------
# Configs & simple containers
# -----------------------------
@dataclass
class StereoCalibConfig:
    model: str = "pinhole"       # kept for parity with your GUI, but rectification here is pinhole
    reference: str = "left"      # 'left' or 'right' as world frame (metadata only)
    cols: int = 9
    rows: int = 6
    square_size_mm: float = 25.0


@dataclass
class StereoResult:
    R: np.ndarray   # rotation RIGHT wrt LEFT (default convention)
    T: np.ndarray   # translation RIGHT wrt LEFT
    E: np.ndarray   # essential
    F: np.ndarray   # fundamental


@dataclass
class LeftFramesValid:
    images: List[np.ndarray]


@dataclass
class RightFramesValid:
    images: List[np.ndarray]


@dataclass
class PairsValid:
    pairs: List[Tuple[np.ndarray, np.ndarray]]


# -----------------------------
# Image pair loaders
# -----------------------------
def _pair_by_sorted_lists(left_paths, right_paths):
    pairs = []
    for lp, rp in zip(sorted(left_paths), sorted(right_paths)):
        L = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        R = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if L is not None and R is not None:
            pairs.append((L, R))
    return pairs  # :contentReference[oaicite:3]{index=3}


def load_image_pairs_smart(folder: str, left_glob="left_*.png", right_glob="right_*.png"):
    left_dir, right_dir = Path(folder, "left"), Path(folder, "right")
    if left_dir.exists() and right_dir.exists():
        return _pair_by_sorted_lists(left_dir.glob("*.*"), right_dir.glob("*.*"))
    return _pair_by_sorted_lists(Path(folder).glob(left_glob), Path(folder).glob(right_glob))  # :contentReference[oaicite:4]{index=4}


# -----------------------------
# Valid-frame filtering
# -----------------------------
def filter_valid_frames(pairs: List[Tuple[np.ndarray, np.ndarray]], cols: int, rows: int):
    pattern = (cols, rows)
    L_valid, R_valid, P_valid, report = [], [], [], []
    for i, (L, R) in enumerate(pairs):
        cL = detect_corners(L, pattern)
        cR = detect_corners(R, pattern)
        okL, okR = cL is not None, cR is not None
        if okL:
            L_valid.append(L)
        if okR:
            R_valid.append(R)
        if okL and okR:
            P_valid.append((L, R))
        report.append({"index": i, "left_ok": okL, "right_ok": okR})
    return LeftFramesValid(L_valid), RightFramesValid(R_valid), PairsValid(P_valid), report  # :contentReference[oaicite:5]{index=5}


# -----------------------------
# Core stereo extrinsics (fixed intrinsics)
# -----------------------------
def stereo_calibrate_from_pairs(
    valid_pairs: PairsValid,
    config: StereoCalibConfig,
    left_intr: CameraResult,
    right_intr: CameraResult,
) -> StereoResult:
    """
    Estimate stereo extrinsics ONLY (R,T,E,F) with FIXED intrinsics from mono calibration.

    Args:
        valid_pairs: PairsValid with frames where both sides see the checkerboard.
        config: chessboard geometry & metadata.
        left_intr/right_intr: CameraResult produced by your mono calibration (K, dist ...).
                              Only K and dist are used here; their R,t are ignored.

    Returns:
        StereoResult (R,T,E,F) with RIGHT w.r.t. LEFT convention.
    """
    if not valid_pairs.pairs:
        raise RuntimeError("No valid pairs")

    pattern = (int(config.cols), int(config.rows))

    # Build 3D object points ONCE (properly scaled in mm)
    objp = prepare_object_points(config.cols, config.rows, config.square_size_mm).astype(np.float32)

    objpoints: List[np.ndarray] = []
    imgL: List[np.ndarray] = []
    imgR: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    for L, R in valid_pairs.pairs:
        cL = detect_corners(L, pattern)
        cR = detect_corners(R, pattern)
        if cL is None or cR is None:
            continue
        h, w = L.shape[:2]
        image_size = (w, h)
        objpoints.append(objp)
        imgL.append(cL.astype(np.float32))
        imgR.append(cR.astype(np.float32))

    if not objpoints or image_size is None:
        raise RuntimeError("No valid detections in pairs")

    # Fix intrinsics from mono (no intrinsics refinement here)
    KL = np.asarray(left_intr.K, dtype=np.float64).reshape(3, 3)
    KR = np.asarray(right_intr.K, dtype=np.float64).reshape(3, 3)
    dL = None if left_intr.dist is None else np.asarray(left_intr.dist, dtype=np.float64).reshape(-1, 1)
    dR = None if right_intr.dist is None else np.asarray(right_intr.dist, dtype=np.float64).reshape(-1, 1)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
    flags = cv2.CALIB_FIX_INTRINSIC  # refine only R,T (and consistent E,F)

    rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgL, imgR,
        KL, dL, KR, dR,
        image_size,
        criteria=criteria,
        flags=flags,
    )

    # Pack and return (don’t store/return monos here by design)
    return StereoResult(
        R=np.asarray(R, float),
        T=np.asarray(T, float),
        E=np.asarray(E, float),
        F=np.asarray(F, float),
    )


# -----------------------------
# Saving: ONLY stereo in the JSON
# -----------------------------
def save_stereo_json(out_path: str, stereo: StereoResult, config: StereoCalibConfig, image_size: Tuple[int, int]):
    """
    Save ONLY the stereo part (R,T,E,F) with minimal metadata.
    Does not write single-camera intrinsics/extrinsics to this file.
    """
    def _arr(a): return np.asarray(a, dtype=float).tolist()
    data = {
        "config": {
            "model": config.model,
            "reference": config.reference,
            "cols": int(config.cols),
            "rows": int(config.rows),
            "square_size_mm": float(config.square_size_mm),
        },
        "image_size": [int(image_size[0]), int(image_size[1])],
        "stereo": {
            "R": _arr(stereo.R),
            "T": _arr(stereo.T),
            "E": _arr(stereo.E),
            "F": _arr(stereo.F),
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
