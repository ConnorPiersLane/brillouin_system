# runtime_camera.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

from brillouin_system.eye_tracker.stereo_calibration.calibration_models import Intrinsics, StereoCalibration


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

    def __init__(self, intr: Intrinsics):
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
class StereoCameras:
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
    def from_stereo_calibration(cls, st: StereoCalibration) -> StereoCameras:
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
        Triangulate 3D point as midpoint of the shortest segment between the
        two viewing rays from left and right cameras.

        - Geometric solution based purely on ray intersection.
        - More robust to calibration noise but not optimal in reprojection error.
        - Falls back to projection from left if rays are nearly parallel.

        Returns
        -------
        np.ndarray
            (3,) world coordinates of the estimated point.
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
        Linear (DLT) triangulation using cv2.triangulatePoints.

        - Works in homogeneous coordinates with camera projection matrices.
        - Minimizes an algebraic error; good as an initial estimate.
        - Assumes P = K [R|t] for each camera and undistorted pixel inputs.

        Returns
        -------
        np.ndarray
            (3,) world coordinates of the triangulated point.
        """
        P1, P2 = self.left.P, self.right.P
        # Undistort to normalized camera coords, then back to pixel homogs with K
        nL = self.left.undistort_points(np.array([uvL]))[0]
        nR = self.right.undistort_points(np.array([uvR]))[0]
        uL = (self.left.K  @ np.array([nL[0], nL[1], 1.0])).reshape(3, 1)
        uR = (self.right.K @ np.array([nR[0], nR[1], 1.0])).reshape(3, 1)
        X_h = cv2.triangulatePoints(P1, P2, uL[:2], uR[:2])  # (4,1)
        return (X_h[:3] / X_h[3]).reshape(3)

    def triangulate_best(self,
                         uvL: Tuple[float, float],
                         uvR: Tuple[float, float],
                         refine: bool = True,
                         max_iters: int = 15,
                         lm_lambda: float = 1e-2) -> Tuple[np.ndarray, float]:
        """
        High-accuracy triangulation combining linear and geometric methods,
        with optional non-linear refinement.

        Steps:
          1. Compute both linear and midpoint estimates.
          2. Choose the one with lower reprojection error.
          3. Optionally refine by minimizing total pixel reprojection error
             in both images (LM optimization).

        Returns
        -------
        Tuple[np.ndarray, float]
            (3,) world coordinates and RMS reprojection error in pixels.
        """

        def project(cam, Xw):
            # cv2.projectPoints expects rvec/tvec mapping WORLD->CAMERA (your cam.R, cam.t already are)
            rvec, _ = cv2.Rodrigues(cam.R)
            dist = cam.dist if cam.dist is not None else np.zeros(5)
            imgpts, _ = cv2.projectPoints(Xw.reshape(1, 1, 3), rvec, cam.t, cam.K, dist)
            return imgpts.reshape(2)

        def reproj_err(Xw):
            pL = project(self.left, Xw);
            eL = pL - np.array(uvL, dtype=np.float64)
            pR = project(self.right, Xw);
            eR = pR - np.array(uvR, dtype=np.float64)
            r = np.hstack([eL, eR])  # [exL, eyL, exR, eyR]
            return r

        def rms(r):
            return float(np.sqrt(np.mean(r ** 2)))

        # 1) candidates
        X_lin = self.triangulate_linear(uvL, uvR)
        X_mid = self.triangulate_midpoint(uvL, uvR)

        r_lin = reproj_err(X_lin);
        rms_lin = rms(r_lin)
        r_mid = reproj_err(X_mid);
        rms_mid = rms(r_mid)

        X = X_lin if rms_lin <= rms_mid else X_mid
        r_best = r_lin if rms_lin <= rms_mid else r_mid

        # Optional: quick cheirality check; if behind a camera, switch seed
        def depths_ok(Xw):
            zL = (self.left.R @ Xw + self.left.t)[2]
            zR = (self.right.R @ Xw + self.right.t)[2]
            return (zL > 0) and (zR > 0)

        if not depths_ok(X):
            X = X_mid if X is X_lin else X_lin
            r_best = reproj_err(X)

        # 2) refine with Gauss-Newton/LM on the 3D point
        if refine:
            eps = 1e-6
            for _ in range(max_iters):
                r = reproj_err(X)
                J = np.zeros((4, 3), dtype=np.float64)
                for k in range(3):
                    d = np.zeros(3);
                    d[k] = eps
                    rp = reproj_err(X + d)
                    J[:, k] = (rp - r) / eps
                # LM step: (JᵀJ + λI) δ = -Jᵀ r
                H = J.T @ J + lm_lambda * np.eye(3)
                g = J.T @ r
                try:
                    delta = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    break
                X_new = X + delta
                # accept if improves reprojection and keeps cheirality
                if depths_ok(X_new) and rms(reproj_err(X_new)) <= rms(r):
                    X = X_new
                else:
                    # damp more if not improving
                    lm_lambda *= 10.0

        return X, rms(reproj_err(X))

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
