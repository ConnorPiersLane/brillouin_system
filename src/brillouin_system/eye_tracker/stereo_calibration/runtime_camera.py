# runtime_camera.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

# Expect these to come from your calib_models.py
# from calib_models import Intrinsics, StereoCalibration


# =========================
# Calibrated single camera
# =========================
class CalibratedCamera:
    """
    Lightweight runtime camera built from Intrinsics.
      - Stores K, dist from Intrinsics
      - WORLD pose (R,t) is set by StereoRig
      - Helpers: undistort_points(), pixel_to_ray_world(), P, C
    """
    __slots__ = ("intr", "K", "dist", "R", "t", "_invK")

    def __init__(self, intr: "Intrinsics"):
        self.intr = intr
        self.K = np.asarray(intr.K, dtype=np.float64).reshape(3, 3)
        self.dist = None if intr.dist is None else np.asarray(intr.dist, dtype=np.float64).reshape(-1)
        # Pose will be filled by StereoRig (defaults here)
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros(3, dtype=np.float64)
        self._invK = np.linalg.inv(self.K)

    # --------- derived basics ---------
    @property
    def C(self) -> np.ndarray:
        """Camera center in WORLD coordinates."""
        return -self.R.T @ self.t

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix (WORLD → image)."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    # --------- core ops ---------
    def undistort_points(self, pts_px: np.ndarray) -> np.ndarray:
        """
        pts_px: (N,2) pixel coordinates in the original image.
        return: (N,2) normalized ideal pinhole coords (x,y) on z=1 plane in camera frame.
        """
        pts_px = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        if self.dist is None:
            uv1 = np.concatenate([pts_px.reshape(-1, 2), np.ones((len(pts_px), 1))], axis=1)  # (N,3)
            rays = (self._invK @ uv1.T).T
            rays /= rays[:, 2:3]
            return rays[:, :2]
        norm = cv2.undistortPoints(pts_px, self.K, self.dist, P=None)  # (N,1,2)
        return norm.reshape(-1, 2)

    def pixel_to_ray_world(self, u: float, v: float) -> np.ndarray:
        """
        Map pixel → unit 3D ray in WORLD coordinates (uses current R).
        """
        xy = self.undistort_points(np.array([[u, v]], dtype=np.float64))[0]
        d_cam = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
        d_cam /= np.linalg.norm(d_cam)
        # camera → world
        return self.R.T @ d_cam


# =========================
# Stereo rig
# =========================
@dataclass
class StereoRig:
    """
    Runtime stereo rig built from StereoCalibration.
    Works for either reference='left' or 'right'.

    Internally we normalize to LEFT→RIGHT:
        X_right = R_lr * X_left + T_lr
    If the saved calibration uses reference='right', we invert it when setting world poses.
    """
    left: CalibratedCamera
    right: CalibratedCamera
    R_lr: np.ndarray
    T_lr: np.ndarray
    reference: str
    image_size: Optional[Tuple[int, int]] = None  # (w,h) convenience for rectification

    # ---- constructor ----
    @classmethod
    def from_stereo_calibration(cls, st: "StereoCalibration") -> "StereoRig":
        camL = CalibratedCamera(st.left)
        camR = CalibratedCamera(st.right)
        # Normalize extrinsics to LEFT→RIGHT, regardless of stored reference
        R_lr, T_lr = st.extr.as_right_wrt_left()
        rig = cls(
            left=camL,
            right=camR,
            R_lr=np.asarray(R_lr, dtype=np.float64).reshape(3, 3),
            T_lr=np.asarray(T_lr, dtype=np.float64).reshape(3),
            reference=st.extr.reference.lower().strip(),
            image_size=st.left.image_size,
        )
        rig._initialize_world_poses()
        return rig

    # ---- set world poses precisely according to the saved reference ----
    def _initialize_world_poses(self):
        """
        If reference='left'  → left at world origin; right placed with (R_lr, T_lr).
        If reference='right' → right at world origin; left placed with inverse (R_rl, T_rl).
        """
        if self.reference == "left":
            # Left is world origin
            self.left.R[:] = np.eye(3); self.left.t[:] = 0.0
            self.right.R[:] = self.left.R @ self.R_lr
            self.right.t[:] = self.left.R @ self.T_lr + self.left.t
        else:
            # Right is world origin: invert LEFT→RIGHT to RIGHT→LEFT
            R_rl = self.R_lr.T
            T_rl = -self.R_lr.T @ self.T_lr
            self.right.R[:] = np.eye(3); self.right.t[:] = 0.0
            self.left.R[:] = self.right.R @ R_rl
            self.left.t[:] = self.right.R @ T_rl + self.right.t

    # ---- triangulation (geometric midpoint) ----
    def triangulate_midpoint(self,
                             uvL: Tuple[float, float],
                             uvR: Tuple[float, float]) -> np.ndarray:
        """
        Midpoint of the shortest segment between the two viewing rays.
        Returns (3,) world coordinates.
        """
        d1 = self.left.pixel_to_ray_world(*uvL)
        d2 = self.right.pixel_to_ray_world(*uvR)
        C1, C2 = self.left.C, self.right.C

        r = C2 - C1
        a = float(d1 @ d1); b = float(d1 @ d2); c = float(d2 @ d2)
        d = float(d1 @ r);  e = float(d2 @ r)
        denom = a * c - b * b
        if abs(denom) < 1e-12:
            # nearly parallel rays; fallback to projecting from left
            s = d / max(a, 1e-12)
            return C1 + s * d1
        s = (d * c - b * e) / denom
        t = (a * e - b * d) / denom
        P1 = C1 + s * d1
        P2 = C2 + t * d2
        return 0.5 * (P1 + P2)

    # ---- triangulation (linear via OpenCV) ----
    def triangulate_linear(self,
                           uvL: Tuple[float, float],
                           uvR: Tuple[float, float]) -> np.ndarray:
        """
        Linear triangulation using cv2.triangulatePoints with projection matrices.
        """
        P1, P2 = self.left.P, self.right.P
        # Undistort to normalized camera coords, then back to pixel homogs with K
        nL = self.left.undistort_points(np.array([uvL]))[0]
        nR = self.right.undistort_points(np.array([uvR]))[0]
        uL = (self.left.K  @ np.array([nL[0], nL[1], 1.0])).reshape(3, 1)
        uR = (self.right.K @ np.array([nR[0], nR[1], 1.0])).reshape(3, 1)
        X_h = cv2.triangulatePoints(P1, P2, uL[:2], uR[:2])  # (4,1)
        return (X_h[:3] / X_h[3]).reshape(3)

    # ---- rectification ----
    def stereo_rectify(self, zero_disparity: bool = True):
        """
        Returns (R1, R2, P1, P2, Q, mapL, mapR) for pinhole model.
        For fisheye lenses, swap in cv2.fisheye.* equivalents as needed.
        """
        if self.image_size is None:
            raise ValueError("image_size is required for rectification")
        flags = cv2.CALIB_ZERO_DISPARITY if zero_disparity else 0
        dl = self.left.dist if self.left.dist is not None else np.zeros(5)
        dr = self.right.dist if self.right.dist is not None else np.zeros(5)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.left.K, dl, self.right.K, dr,
            self.image_size, self.R_lr, self.T_lr, flags=flags
        )
        mapL = cv2.initUndistortRectifyMap(self.left.K,  dl, R1, P1, self.image_size, cv2.CV_32FC1)
        mapR = cv2.initUndistortRectifyMap(self.right.K, dr, R2, P2, self.image_size, cv2.CV_32FC1)
        return (R1, R2, P1, P2, Q, mapL, mapR)
