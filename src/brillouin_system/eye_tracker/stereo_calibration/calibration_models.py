from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import json
import numpy as np
import cv2


# ----------------- small helpers -----------------
def _np(a, shape=None, dtype=float) -> np.ndarray:
    x = np.asarray(a, dtype=dtype)
    return x.reshape(shape) if shape else x

def _zeros_like_dist(dist_len: int) -> np.ndarray:
    return np.zeros((dist_len,), dtype=float)

def _skew(t: np.ndarray) -> np.ndarray:
    tx, ty, tz = t.reshape(3)
    return np.array([[0, -tz,  ty],
                     [tz,  0, -tx],
                     [-ty, tx,  0]], float)


# ----------------- core dataclasses -----------------
@dataclass
class Intrinsics:
    """
    Minimal mono model:
      - K (3x3)
      - dist (N,)   (use OpenCV's standard order for 'pinhole': k1,k2,p1,p2,k3[,k4,k5,k6])
      - image_size: (w,h)
      - model: 'pinhole' or 'fisheye'
      - mono_rms: optional reprojection RMS from mono calibration
    """
    K: np.ndarray
    dist: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None
    model: str = "pinhole"
    mono_rms: Optional[float] = None

    def __post_init__(self):
        self.K = _np(self.K, (3, 3))
        self.dist = None if self.dist is None else _np(self.dist).reshape(-1)
        if self.image_size is not None:
            self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.model = self.model.lower().strip()

    # --- convenience ---
    def fov_deg(self) -> Tuple[float, float]:
        """(hfov, vfov) in degrees based on K and image_size (approx)."""
        if not self.image_size:
            return (float("nan"), float("nan"))
        w, h = self.image_size
        fx, fy = self.K[0, 0], self.K[1, 1]
        hfov = 2.0 * np.degrees(np.arctan2(w, 2.0 * fx))
        vfov = 2.0 * np.degrees(np.arctan2(h, 2.0 * fy))
        return (hfov, vfov)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "K": self.K.tolist(),
            "dist": None if self.dist is None else self.dist.tolist(),
            "image_size": None if self.image_size is None else list(self.image_size),
            "model": self.model,
            "mono_rms": None if self.mono_rms is None else float(self.mono_rms),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Intrinsics":
        return cls(
            K=d["K"],
            dist=d.get("dist", None),
            image_size=None if d.get("image_size") is None else tuple(d["image_size"]),
            model=d.get("model", "pinhole"),
            mono_rms=d.get("mono_rms", None),
        )


@dataclass
class StereoExtrinsics:
    """
    Relative pose RIGHT wrt LEFT by default:
        X_right = R * X_left + T
    'reference' = 'left' or 'right' indicates which camera is the world frame.
    """
    R: np.ndarray
    T: np.ndarray
    reference: str = "left"

    def __post_init__(self):
        self.R = _np(self.R, (3, 3))
        self.T = _np(self.T, (3,))
        self.reference = self.reference.lower().strip()

    # normalized accessors
    def as_right_wrt_left(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.reference == "left":
            return self.R, self.T
        # stored as right-ref; invert
        return self.R.T, -self.R.T @ self.T

    def as_left_wrt_right(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.reference == "right":
            return self.R, self.T
        return self.R.T, -self.R.T @ self.T

    def to_dict(self) -> Dict[str, Any]:
        return {"R": self.R.tolist(), "T": self.T.tolist(), "reference": self.reference}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StereoExtrinsics":
        return cls(R=d["R"], T=d["T"], reference=d.get("reference", "left"))


@dataclass
class StereoCalibration:
    """
    Everything you need for stereo rectification & depth:
      - left/right Intrinsics
      - extr: StereoExtrinsics
      - stereo_rms: reprojection RMS from stereoCalibrate
    E and F are derived on demand (no duplication on disk).
    """
    left: Intrinsics
    right: Intrinsics
    extr: StereoExtrinsics
    stereo_rms: Optional[float] = None

    # --- derived quantities ---
    def essential(self) -> np.ndarray:
        R, T = self.extr.as_right_wrt_left()
        return _skew(T) @ R

    def fundamental(self) -> np.ndarray:
        R, T = self.extr.as_right_wrt_left()
        K1, K2 = self.left.K, self.right.K
        E = _skew(T) @ R
        return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    # --- rectification maps (OpenCV) ---
    def stereo_rectify(self, flags=cv2.CALIB_ZERO_DISPARITY):
        if self.left.image_size is None:
            raise ValueError("left.image_size is required for rectification")
        w, h = self.left.image_size
        R, T = self.extr.as_right_wrt_left()

        # choose dist shapes
        if self.left.model == "fisheye":
            # For fisheye, use cv2.fisheye.initUndistortRectifyMap instead if needed by your pipeline.
            raise NotImplementedError("Fisheye rectification not included in minimal model.")
        dl = self.left.dist if self.left.dist is not None else _zeros_like_dist(5)
        dr = self.right.dist if self.right.dist is not None else _zeros_like_dist(5)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.left.K, dl, self.right.K, dr, (w, h), R, T, flags=flags
        )
        mapL = cv2.initUndistortRectifyMap(self.left.K, dl, R1, P1, (w, h), cv2.CV_32FC1)
        mapR = cv2.initUndistortRectifyMap(self.right.K, dr, R2, P2, (w, h), cv2.CV_32FC1)
        return (R1, R2, P1, P2, Q, mapL, mapR)

    # --- JSON I/O ---
    def to_dict(self) -> Dict[str, Any]:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "stereo": self.extr.to_dict(),
            "stereo_rms": None if self.stereo_rms is None else float(self.stereo_rms),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StereoCalibration":
        return cls(
            left=Intrinsics.from_dict(d["left"]),
            right=Intrinsics.from_dict(d["right"]),
            extr=StereoExtrinsics.from_dict(d["stereo"]),
            stereo_rms=d.get("stereo_rms", None),
        )

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "StereoCalibration":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ----------------- simple builders from OpenCV outputs -----------------
def intrinsics_from_opencv(
    K: np.ndarray,
    dist: Optional[np.ndarray],
    image_size: Tuple[int, int],
    model: str = "pinhole",
    mono_rms: Optional[float] = None,
) -> Intrinsics:
    return Intrinsics(K=K, dist=None if dist is None else dist.reshape(-1),
                      image_size=(int(image_size[0]), int(image_size[1])),
                      model=model, mono_rms=mono_rms)

def stereo_from_opencv(
    left: Intrinsics,
    right: Intrinsics,
    R: np.ndarray,
    T: np.ndarray,
    reference: str = "left",
    stereo_rms: Optional[float] = None,
) -> StereoCalibration:
    extr = StereoExtrinsics(R=R, T=T, reference=reference)
    return StereoCalibration(left=left, right=right, extr=extr, stereo_rms=stereo_rms)
