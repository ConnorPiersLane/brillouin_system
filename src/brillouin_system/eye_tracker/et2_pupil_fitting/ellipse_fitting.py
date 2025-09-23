# pupil_pipeline.py
# Dependencies: pip install opencv-python numpy

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
import numpy as np
import cv2

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Ellipse2D:
    """
    An ellipse fitted in image pixel coordinates.

    Attributes
    ---------
    cx, cy : float
        Center of the ellipse (pixels).
    major, minor : float
        Major/minor *full* axis lengths (pixels) as returned by cv2.fitEllipse.
    angle_deg : float
        Rotation angle of the ellipse in degrees (OpenCV convention).
    """
    cx: float
    cy: float
    major: float
    minor: float
    angle_deg: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def axes(self) -> Tuple[float, float]:
        return (self.major, self.minor)

    @property
    def axis_ratio(self) -> float:
        a = max(self.major, self.minor) * 0.5
        b = min(self.major, self.minor) * 0.5
        return float(a / (b + 1e-12))

    @property
    def area(self) -> float:
        return float(np.pi * (self.major * 0.5) * (self.minor * 0.5))

    def to_opencv(self) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Return OpenCV-style ellipse tuple."""
        return ((self.cx, self.cy), (self.major, self.minor), self.angle_deg)

    @staticmethod
    def from_opencv(e: Tuple[Tuple[float, float], Tuple[float, float], float]) -> "Ellipse2D":
        (cx, cy), (MA, ma), ang = e
        return Ellipse2D(float(cx), float(cy), float(MA), float(ma), float(ang))


@dataclass
class PupilFitConfig:
    """
    Minimal, robust knobs for fast pupil ellipse fitting.
    Keep it small so you rarely need to touch it.
    """
    # Preprocessing
    gaussian_ksize: int = 5             # must be odd; 5 is a good default
    # Threshold: pupil is darker than iris/sclera -> invert + Otsu
    use_otsu: bool = True
    # Morphology
    close_kernel: int = 3               # 3x3 elliptical kernel
    close_iterations: int = 1
    # Contour gating
    min_area_frac: float = 5e-4         # reject too small blobs (fraction of processed image area)
    max_area_frac: float = 0.5          # reject too large blobs
    max_bbox_aspect: float = 3.5        # reject extreme skinny blobs
    # Downscaling for speed (rescaled back to original coords)
    scale: float = 0.5                  # 0.4–0.6 is typically great
    # Optional ROI (x,y,w,h). Use when you can crop to the eye region for speed & robustness.
    roi: Optional[Tuple[int, int, int, int]] = None


@dataclass
class CalibPara:
    """
    Calibrated camera parameters (WORLD -> CAMERA). Pure data holder; safe to serialize.

    K : (3,3) intrinsics
    R : (3,3) rotation (world->camera)
    t : (3,)  translation (world->camera)
    dist : (n,) distortion coefficients (optional; standard OpenCV pinhole order)
    """
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: Optional[np.ndarray] = None


class CalibratedCamera:
    """
    Fast runtime camera built from CalibPara or raw arrays.
    Validates, precomputes, and exposes helpers (undistort, pixel->ray, triangulate).
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
        self.C = -self.R.T @ self.t                                  # camera center in WORLD

    @classmethod
    def from_params(cls, p: CalibPara) -> "CalibratedCamera":
        return cls(p.K, p.R, p.t, p.dist)

    # ---- helpers ----
    def undistort_points(self, pts_px: np.ndarray) -> np.ndarray:
        """
        pts_px: (N,2) pixel coords from the original image.
        Returns: (N,2) normalized (x,y) coords on the z=1 plane in camera frame.
        """
        pts_px = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        if self.dist is None:
            homog = np.concatenate([pts_px.reshape(-1, 2), np.ones((len(pts_px), 1))], axis=1)  # (N,3)
            norm = (self._invK @ homog.T).T
            norm /= norm[:, 2:3]
            return norm[:, :2]
        norm = cv2.undistortPoints(pts_px, self.K, self.dist, P=None)  # (N,1,2)
        return norm.reshape(-1, 2)

    def pixel_to_ray_world(self, u: float, v: float) -> np.ndarray:
        """Map a pixel to a unit 3D ray direction in *world* coordinates."""
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

# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

class PupilFitter:
    """One-pass, performant pupil ellipse fitter."""
    def __init__(self, cfg: Optional[PupilFitConfig] = None):
        self.cfg = cfg or PupilFitConfig()

    def fit(self, image: np.ndarray) -> Optional[Ellipse2D]:
        """
        Fit the pupil ellipse from an image.

        Returns
        -------
        Ellipse2D or None
        """
        cfg = self.cfg
        gray_full = _to_gray(image)
        H, W = gray_full.shape[:2]

        # ROI crop (recommended if available)
        if cfg.roi is not None:
            x, y, w, h = cfg.roi
            x = max(0, x); y = max(0, y)
            w = min(w, W - x); h = min(h, H - y)
            gray = gray_full[y:y+h, x:x+w]
            offx, offy = x, y
        else:
            gray = gray_full
            offx = offy = 0

        # Optional downscale for speed
        if 0.0 < cfg.scale < 1.0:
            gray_small = cv2.resize(gray, None, fx=cfg.scale, fy=cfg.scale, interpolation=cv2.INTER_AREA)
        else:
            gray_small = gray

        # Light blur
        k = int(cfg.gaussian_ksize)
        if k < 3: k = 3
        if k % 2 == 0: k += 1  # ensure odd
        blur = cv2.GaussianBlur(gray_small, (k, k), 0)

        # Threshold (pupil dark → invert)
        if cfg.use_otsu:
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            thr = float(np.median(blur) - 15.0)
            thr = max(0.0, thr)
            _, mask = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY_INV)

        # Close small gaps
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        if cfg.close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=int(cfg.close_iterations))

        # Contours
        cnts_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
        if not contours:
            return None

        img_area = float(mask.shape[0] * mask.shape[1])
        best = None; best_area = 0.0

        for c in contours:
            if len(c) < 5:
                continue  # fitEllipse needs >=5 points
            area = cv2.contourArea(c)
            if area < self.cfg.min_area_frac * img_area or area > self.cfg.max_area_frac * img_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = max(w, h) / (min(w, h) + 1e-9)
            if ar > self.cfg.max_bbox_aspect:
                continue
            if area > best_area:
                best_area = area
                best = c

        if best is None:
            return None

        (cx, cy), (MA, ma), ang = cv2.fitEllipse(best)

        # Upscale + shift back to original coords
        if 0.0 < cfg.scale < 1.0:
            cx /= cfg.scale; cy /= cfg.scale
            MA /= cfg.scale; ma /= cfg.scale
        cx += offx; cy += offy

        # Ensure (major, minor) ordering
        if ma > MA:
            MA, ma = ma, MA
            ang = (ang + 90.0) % 180.0

        return Ellipse2D(cx=float(cx), cy=float(cy),
                         major=float(MA), minor=float(ma),
                         angle_deg=float(ang))

# ----------------------------------------------------------------------
# Conic & plane utilities (for perspective-correct centers)
# ----------------------------------------------------------------------

def ellipse_to_conic(e: Ellipse2D) -> np.ndarray:
    """
    Convert Ellipse2D to 3x3 image conic Q (primal form), s.t. x^T Q x = 0 for boundary points x~(u,v,1).
    """
    (cx, cy), (MA, ma), angle = e.center, e.axes, e.angle_deg
    a = float(MA) / 2.0
    b = float(ma) / 2.0
    theta = np.deg2rad(float(angle))

    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s],
                   [s,  c]], dtype=np.float64)
    S2 = np.diag([a, b])
    A = R2 @ S2
    H = np.array([[A[0,0], A[0,1], cx],
                  [A[1,0], A[1,1], cy],
                  [0,      0,      1 ]], dtype=np.float64)
    C0 = np.diag([1.0, 1.0, -1.0])  # unit circle conic
    Hinv = np.linalg.inv(H)
    Q = Hinv.T @ C0 @ Hinv
    return 0.5 * (Q + Q.T)

def build_view_cone(P: np.ndarray, Q_img: np.ndarray) -> np.ndarray:
    """
    Lift an image ellipse (Q_img) into a 3D *view cone* quadric: C = P^T Q_img P  (4x4).
    """
    return P.T @ Q_img @ P

def adjugate_4x4(A: np.ndarray) -> np.ndarray:
    """
    Adjugate (cofactor transpose) of a 4x4 matrix.
    Works for singular quadrics (cones) where inverse may not exist.
    """
    A = np.asarray(A, dtype=np.float64).reshape(4, 4)
    cof = np.zeros((4, 4), dtype=np.float64)
    # Compute cofactors C_ij = (-1)^{i+j} det(M_ij)
    idx = [0,1,2,3]
    for i in range(4):
        for j in range(4):
            rows = [r for r in idx if r != i]
            cols = [c for c in idx if c != j]
            M = A[np.ix_(rows, cols)]
            cof[i, j] = ((-1.0)**(i+j)) * np.linalg.det(M)
    return cof.T  # adj(A) = C^T

def inside_sign(Q: np.ndarray, center_uv: Tuple[float,float]) -> float:
    """
    Determine the sign convention for 'inside' of an ellipse conic Q by evaluating at its center.
    """
    u, v = center_uv
    x = np.array([u, v, 1.0], dtype=np.float64)
    val = float(x @ Q @ x)
    # Inside should be opposite sign of boundary (0). Use sign at center as inside.
    return -1.0 if val > 0 else 1.0

def point_in_ellipse(Q: np.ndarray, uv: Tuple[float,float], sign_inside: float) -> bool:
    u, v = uv
    x = np.array([u, v, 1.0], dtype=np.float64)
    return sign_inside * float(x @ Q @ x) < 0.0

# ----------------------------------------------------------------------
# 3D pupil center — two paths
# ----------------------------------------------------------------------

def pupil_center_3d_from_two_ellipses_centers(
    e1: Ellipse2D, cam1: CalibratedCamera,
    e2: Ellipse2D, cam2: CalibratedCamera,
    warn_on_eccentric: bool = True
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Simple fallback: triangulate the ellipse centers (weak-perspective assumption).
    """
    p1 = e1.center
    p2 = e2.center
    if warn_on_eccentric and max(e1.axis_ratio, e2.axis_ratio) > 1.20:
        print("[pupil_pipeline] Warning: high ellipse eccentricity "
              f"(r1={e1.axis_ratio:.2f}, r2={e2.axis_ratio:.2f}). "
              "Triangulating centers may be biased; consider perspective-correct centers.")

    P = cam1.triangulate_midpoint(p1, cam2, p2)
    return P, p1, p2


def pupil_center_3d_from_two_ellipses_pc(
    e1: Ellipse2D, cam1: CalibratedCamera,
    e2: Ellipse2D, cam2: CalibratedCamera,
    *,
    return_intermediates: bool = False
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float], dict]:
    """
    Perspective-correct 3D pupil center from two image ellipses.
    No plane normal required. Steps:
      1) Build image conics Q1,Q2.
      2) Lift to 3D view-cones C1=P1^T Q1 P1, C2=P2^T Q2 P2 (4x4).
      3) Estimate the pupil plane Π (4,) by minimizing the dual tangency residual:
             r_i(Π) = Π^T C_i^* Π   with  C_i^* = adj(C_i)
         We take Π as the smallest-eigenvector of M = C1*^2 + C2*^2 (symmetrized),
         and evaluate a few smallest-eig candidates; pick the one that yields centers
         inside both ellipses.
      4) For each image, vanishing line ℓ_i ∝ P_i Π; true projected center p_i ∝ Q_i^{-1} ℓ_i.
      5) Triangulate p1, p2.

    Returns:
        P_world, p1, p2, info_dict
    """
    # 1) Image conics
    Q1 = ellipse_to_conic(e1)
    Q2 = ellipse_to_conic(e2)

    # 2) 3D view-cones
    C1 = build_view_cone(cam1.P, Q1)
    C2 = build_view_cone(cam2.P, Q2)

    # 3) Duals (adjugates), then find plane Π minimizing sum of squared tangency
    C1_star = adjugate_4x4(C1)
    C2_star = adjugate_4x4(C2)

    # Form symmetric M = C1*^2 + C2*^2
    M = C1_star @ C1_star + C2_star @ C2_star
    M = 0.5 * (M + M.T)

    # Candidates: k smallest eigenvectors
    evals, evecs = np.linalg.eigh(M)
    order = np.argsort(evals)
    candidates = [evecs[:, order[i]] for i in range(min(3, len(order)))]  # up to 3 best

    # ellipse "inside" sign convention (use center)
    s1 = inside_sign(Q1, e1.center)
    s2 = inside_sign(Q2, e2.center)

    chosen = None
    best_score = np.inf
    best = {}

    for pi in candidates:
        # Normalize plane vector for stability
        pi = pi / (np.linalg.norm(pi[:3]) + 1e-12)

        # 4) Vanishing lines and true projected centers
        l1 = cam1.P @ pi  # (3,)
        l2 = cam2.P @ pi
        # Solve Q x = l  -> x = Q^{-1} l
        try:
            p1_h = np.linalg.solve(Q1, l1)
            p2_h = np.linalg.solve(Q2, l2)
        except np.linalg.LinAlgError:
            continue
        if abs(p1_h[2]) < 1e-12 or abs(p2_h[2]) < 1e-12:
            continue
        p1 = (float(p1_h[0] / p1_h[2]), float(p1_h[1] / p1_h[2]))
        p2 = (float(p2_h[0] / p2_h[2]), float(p2_h[1] / p2_h[2]))

        # Check inside-ellipse and residuals
        ok1 = point_in_ellipse(Q1, p1, s1)
        ok2 = point_in_ellipse(Q2, p2, s2)
        r1 = float(pi @ (C1_star @ pi))
        r2 = float(pi @ (C2_star @ pi))
        score = abs(r1) + abs(r2) + (0.0 if (ok1 and ok2) else 1e3)

        if score < best_score:
            best_score = score
            chosen = (pi, p1, p2, r1, r2)

    # Fallback to centers if no candidate worked
    if chosen is None:
        P_fallback, p1c, p2c = pupil_center_3d_from_two_ellipses_centers(e1, cam1, e2, cam2)
        info = dict(method="fallback_centers", residual_sum=np.nan)
        return P_fallback, p1c, p2c, info

    pi, p1, p2, r1, r2 = chosen

    # 5) Triangulate the corrected centers
    P_world = cam1.triangulate_midpoint(p1, cam2, p2)
    info = dict(method="plane_from_dual_cones", plane=pi, residuals=(r1, r2))
    return P_world, p1, p2, info

# ----------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------

def fit_ellipse_step(image: np.ndarray, cfg: Optional[PupilFitConfig] = None) -> Optional[Ellipse2D]:
    """One clean step: fit the pupil ellipse from an image, using a tiny config."""
    return PupilFitter(cfg).fit(image)

def draw_ellipse(image: np.ndarray, e: Optional[Ellipse2D],
                 color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Return a copy with the ellipse drawn (no side effects)."""
    if e is None:
        return image.copy()
    out = image.copy()
    cv2.ellipse(out, e.to_opencv(), color, thickness)
    cv2.circle(out, (int(round(e.cx)), int(round(e.cy))), 2, color, -1)
    return out

# ----------------------------------------------------------------------
# Example (commented)
# ----------------------------------------------------------------------
"""
# Example usage:

# 1) Fit per view
cfg = PupilFitConfig(scale=0.5, roi=(x, y, w, h))  # use ROI if you have an eye box
eL = fit_ellipse_step(img_left, cfg)
eR = fit_ellipse_step(img_right, cfg)
if eL is None or eR is None:
    raise RuntimeError("Failed to fit ellipse(s).")

# 2) Cameras (world->cam)
camL = CalibratedCamera(K=K_L, R=R_L, t=t_L, dist=dist_L)  # dist optional
camR = CalibratedCamera(K=K_R, R=R_R, t=t_R, dist=dist_R)

# 3) Perspective-correct 3D pupil center (no plane normal required)
P_world, pL, pR, info = pupil_center_3d_from_two_ellipses_pc(eL, camL, eR, camR)
print("3D pupil center:", P_world, "method:", info["method"], "residuals:", info["residuals"])

# (Optional) simple fallback:
# P_fallback, pL0, pR0 = pupil_center_3d_from_two_ellipses_centers(eL, camL, eR, camR)
"""
