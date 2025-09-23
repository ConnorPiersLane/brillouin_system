# calibration.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
import numpy as np
import cv2

# -------------------- Persisted POD (no logic) --------------------

@dataclass
class CalibPara:
    """
    Calibrated camera parameters (WORLD -> CAMERA).
    Pure data holder: safe to save/load with your existing HDF5 pipeline.

    K : (3,3) intrinsics
    R : (3,3) rotation (world->camera)
    t : (3,)  translation (world->camera)
    dist : (n,) distortion coefficients (optional; standard OpenCV pinhole order)
    """
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: Optional[np.ndarray] = None


# -------------------- Runtime wrapper (fast helpers) --------------------

class CalibratedCamera:
    """
    Fast runtime camera built from CalibPara or raw arrays.
    Does all validation/normalization on construction, then exposes helpers.

    Notes:
    - Does *not* mutate the provided CalibPara (copies/coerces internally).
    - Assumes standard OpenCV pinhole distortion if `dist` is provided.
      (For fisheye, pre-undistort points or add a small branch.)
    """
    __slots__ = ("K", "R", "t", "dist", "P", "C", "_invK")

    # ---- constructors ----
    def __init__(self, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: Optional[np.ndarray] = None):
        self.K   = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.R   = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t   = np.asarray(t, dtype=np.float64).reshape(3)
        if dist is None or (isinstance(dist, Iterable) and len(dist) == 0):
            self.dist = None
        else:
            self.dist = np.asarray(dist, dtype=np.float64).reshape(-1)

        # precompute
        self._invK = np.linalg.inv(self.K)
        self.P = self.K @ np.hstack([self.R, self.t.reshape(3, 1)])  # 3x4 projection
        self.C = -self.R.T @ self.t  # camera center in WORLD

    @classmethod
    def from_params(cls, p: CalibPara) -> "CalibratedCamera":
        """Build a runtime camera from a CalibPara POD."""
        return cls(p.K, p.R, p.t, p.dist)

    # ---- helpers ----
    def undistort_points(self, pts_px: np.ndarray) -> np.ndarray:
        """
        pts_px: (N,2) pixel coords from the original image.
        Returns: (N,2) normalized (x,y) coords on the z=1 plane in camera frame.
        """
        pts_px = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        if self.dist is None:
            # No distortion -> just apply invK
            homog = np.concatenate([pts_px.reshape(-1, 2), np.ones((len(pts_px), 1))], axis=1)  # (N,3)
            norm = (self._invK @ homog.T).T
            norm /= norm[:, 2:3]
            return norm[:, :2]
        norm = cv2.undistortPoints(pts_px, self.K, self.dist, P=None)  # (N,1,2)
        return norm.reshape(-1, 2)

    def pixel_to_ray_world(self, u: float, v: float) -> np.ndarray:
        """
        Map a pixel to a unit 3D ray direction in *world* coordinates.
        """
        if self.dist is None:
            d_cam = self._invK @ np.array([u, v, 1.0], dtype=np.float64)
        else:
            x, y = self.undistort_points(np.array([[u, v]], dtype=np.float64))[0]
            d_cam = np.array([x, y, 1.0], dtype=np.float64)
        d_cam /= np.linalg.norm(d_cam)
        return self.R.T @ d_cam  # camera->world

    def triangulate_midpoint(self,
                             uv1: Tuple[float, float],
                             other: "CalibratedCamera",
                             uv2: Tuple[float, float]) -> np.ndarray:
        """
        Midpoint of closest approach between this camera's ray through uv1
        and the other's ray through uv2. Returns (3,) world coords.
        """
        d1 = self.pixel_to_ray_world(*uv1)
        d2 = other.pixel_to_ray_world(*uv2)
        C1, C2 = self.C, other.C
        r = C2 - C1
        a = float(d1 @ d1); b = float(d1 @ d2); c = float(d2 @ d2)
        d = float(d1 @ r);  e = float(d2 @ r)
        denom = a * c - b * b
        if abs(denom) < 1e-12:
            s = d / max(a, 1e-12)
            return C1 + s * d1
        s = (d * c - b * e) / denom
        t = (a * e - b * d) / denom
        P1 = C1 + s * d1
        P2 = C2 + t * d2
        return 0.5 * (P1 + P2)
