from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

from brillouin_system.eye_tracker.stereo_imaging.calibration_dataclasses import Intrinsics, StereoCalibration
from brillouin_system.eye_tracker.stereo_imaging.njit_helpers import _lm_refine_point


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

    INTERNAL CONVENTION:
      - WORLD = LEFT camera (always).
      - Baseline normalized to LEFT→RIGHT:
            X_right = R_lr * X_left + T_lr
    """
    st_cal: StereoCalibration
    left: CalibratedCamera
    right: CalibratedCamera
    R_lr: np.ndarray
    T_lr: np.ndarray
    image_size: tuple[int, int] | None = None  # (w,h) convenience for rectification

    # ---- constructor ----
    @classmethod
    def from_stereo_calibration(cls, st: StereoCalibration) -> StereoCameras:
        camL = CalibratedCamera(st.left)
        camR = CalibratedCamera(st.right)
        # Normalize extrinsics to LEFT→RIGHT, regardless of stored reference
        R_lr, T_lr = st.extr.as_right_wrt_left()
        rig = cls(
            st_cal=st,
            left=camL,
            right=camR,
            R_lr=np.asarray(R_lr, dtype=np.float64).reshape(3, 3),
            T_lr=np.asarray(T_lr, dtype=np.float64).reshape(3),
            image_size=st.left.image_size,
        )
        rig._initialize_world_poses()
        return rig

    def _initialize_world_poses(self):
        """
        Force WORLD = LEFT camera for all cases.
        Right pose is placed via the normalized LEFT→RIGHT extrinsics (R_lr, T_lr).
        """
        # Left is the world origin
        self.left.R[:] = np.eye(3, dtype=np.float64)
        self.left.t[:] = 0.0

        # Right expressed in the LEFT/world frame
        self.right.R[:] = self.R_lr
        self.right.t[:] = self.T_lr
        self.prepare_projection()

    def prepare_projection(self):
        # rvec/tvec for cv2.projectPoints (WORLD->CAM)
        self._rvecL, _ = cv2.Rodrigues(self.left.R.astype(np.float64))
        self._rvecR, _ = cv2.Rodrigues(self.right.R.astype(np.float64))
        self._tvecL = self.left.t.astype(np.float64).reshape(3, 1)
        self._tvecR = self.right.t.astype(np.float64).reshape(3, 1)
        # full projection matrices (for DLT)
        self._P1 = self.left.P
        self._P2 = self.right.P
        # cached zero-dist if None
        self._dl = self.left.dist if self.left.dist is not None else np.zeros(5)
        self._dr = self.right.dist if self.right.dist is not None else np.zeros(5)


    # ---- triangulation (geometric midpoint) ----
    def triangulate_midpoint(self,
                             uvL: tuple[float, float],
                             uvR: tuple[float, float]) -> np.ndarray:
        """
        Triangulate 3D point as midpoint of the shortest segment between the
        two viewing rays from left and right cameras.

        - Geometric solution based purely on ray intersection.
        - More robust to calibration noise but not optimal in reprojection error.
        - Falls back to projection from left if rays are nearly parallel.

        Returns
        -------
        np.ndarray
            (3,) left camera coordinates of the estimated point.
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
    def triangulate_linear(self, uvL, uvR):
        P1, P2 = self._P1, self._P2
        # undistort to normalized, then back to pixel homogs with K
        nL = self.left.undistort_points(np.array([uvL]))[0]
        nR = self.right.undistort_points(np.array([uvR]))[0]
        uL = (self.left.K @ np.array([nL[0], nL[1], 1.0])).reshape(3, 1)
        uR = (self.right.K @ np.array([nR[0], nR[1], 1.0])).reshape(3, 1)
        X_h = cv2.triangulatePoints(P1, P2, uL[:2], uR[:2])
        return (X_h[:3] / X_h[3]).reshape(3)

    def _project(self, cam, Xw):
        rvec = self._rvecL if cam is self.left else self._rvecR
        tvec = self._tvecL if cam is self.left else self._tvecR
        dist = self._dl if cam is self.left else self._dr
        imgpts, _ = cv2.projectPoints(Xw.reshape(1, 1, 3), rvec, tvec, cam.K, dist)
        return imgpts.reshape(2)

    def _reproj_err(self, Xw, uvL, uvR):
        eL = self._project(self.left, Xw) - np.array(uvL, dtype=np.float64)
        eR = self._project(self.right, Xw) - np.array(uvR, dtype=np.float64)
        return np.hstack([eL, eR])

    @staticmethod
    def _rms(r):
        return float(np.sqrt(np.mean(r * r)))


    def triangulate_best(self, uvL, uvR, refine=True, max_iters=5, lm_lambda=1e-2, refine_thresh_px=0.5):

        X_lin = self.triangulate_linear(uvL, uvR)
        X_mid = self.triangulate_midpoint(uvL, uvR)
        rms_lin = self._rms(self._reproj_err(X_lin, uvL, uvR))
        rms_mid = self._rms(self._reproj_err(X_mid, uvL, uvR))
        X = X_lin if rms_lin <= rms_mid else X_mid
        rms_curr = min(rms_lin, rms_mid)

        def depths_ok(Xw):
            zL = (self.left.R @ Xw + self.left.t)[2]
            zR = (self.right.R @ Xw + self.right.t)[2]
            return (zL > 0) and (zR > 0)

        if not depths_ok(X):
            X = X_mid if np.allclose(X, X_lin) else X_lin
            rms_curr = self._rms(self._reproj_err(X, uvL, uvR))

        if not refine or rms_curr <= refine_thresh_px:
            return X, rms_curr

        # inputs for numba function (use already-cached members; shapes must be 2D for njit)
        K_L = self.left.K
        K_R = self.right.K
        dist_L = self._dl if self.left.dist is not None else np.zeros(5, dtype=np.float64)
        dist_R = self._dr if self.right.dist is not None else np.zeros(5, dtype=np.float64)
        R_L = self.left.R
        R_R = self.right.R
        t_L = self.left.t
        t_R = self.right.t
        uvL_u = np.array(uvL, dtype=np.float64)
        uvR_u = np.array(uvR, dtype=np.float64)

        if rms_curr < 1e-9:  # tiny error → LM unstable / unnecessary
            return X, rms_curr

        X_refined, rms_refined = _lm_refine_point(
            X, K_L, dist_L, R_L, t_L, K_R, dist_R, R_R, t_R, uvL_u, uvR_u,
            max_iters=max_iters, eps=1e-6, lam0=lm_lambda,
            tol_step=1e-8, tol_improve=1e-6
        )
        return X_refined, rms_refined

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
