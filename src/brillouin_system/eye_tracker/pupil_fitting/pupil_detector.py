# pupil_detector.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter import EllipseFitter
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitting_helpers import ellipse_to_conic, build_view_cone, \
    adjugate_4x4, inside_sign, point_in_ellipse
from brillouin_system.eye_tracker.stereo_imaging.se3 import SE3
from brillouin_system.eye_tracker.stereo_imaging.stereo_cameras import StereoCameras


# If you want to default-load your calibrated stereo rig:
# from brillouin_system.eye_tracker.stereo_calibration.init_stereo_cameras import stereo_cameras

@dataclass
class Pupil3D:
    """Container for 3D pupil results."""
    center_left: np.ndarray | None   # (3,) in LEFT camera frame
    center_ref:  np.ndarray | None  # (3,) in reference frame (via SE3)
    normal_left: np.ndarray | None   # (3,) optional pupil normal in LEFT
    normal_ref:  np.ndarray | None   # (3,) optional pupil normal in REF


class PupilDetector:
    """
    Composes:
      - EllipseFitter (2D pupil detection in each image)
      - StereoCameras (triangulation from L/R)
      - SE3 transform (LEFT -> reference frame, e.g., 'zaber')

    Minimal responsibilities for now:
      * store dependencies
      * provide accessors for current reference frame
      * (stubs) detect/triangulate methods to be filled next
    """

    def __init__(
        self,
        ellipse_fitter: EllipseFitter,
        stereo_cameras: StereoCameras,                      # type:
        left_to_ref: SE3 = None,   # LEFT -> REF; if None, identity (left)
        ref_name: str = "left",
    ) -> None:
        self.ellipse_fitter = ellipse_fitter
        self.stereo = stereo_cameras
        self.ref_name = ref_name

        if left_to_ref is None:
            self.T_left_to_ref = SE3(np.eye(3), np.zeros(3))
            self.ref_name = "left"
        else:
            self.T_left_to_ref = left_to_ref

    # ---------------- convenience ----------------
    def set_reference(self, T_left_to_ref: SE3, name: str = "ref") -> None:
        """Update the output reference frame (LEFT -> REF)."""
        self.T_left_to_ref = T_left_to_ref
        self.ref_name = name

    def get_reference_name(self) -> str:
        return self.ref_name

    # ---------------- stubs you can fill next ----------------
    def find_pupil_left_2d(self, img_left) -> Ellipse2D:
        """Detect pupil ellipse in the LEFT image (2D)."""
        return self.ellipse_fitter.find_pupil_left(img_left)

    def find_pupil_right_2d(self, img_right) -> Ellipse2D:
        """Detect pupil ellipse in the RIGHT image (2D)."""
        return self.ellipse_fitter.find_pupil_right(img_right)

    def triangulate_center(
        self,
        eL: Ellipse2D,
        eR: Ellipse2D,
    ) -> Pupil3D:
        """
        Simple baseline: triangulate the image-ellipse centers (weak perspective).
        Uses the rig's robust triangulator and then maps to the selected reference via SE3.
        """
        if eL is None or eR is None:
            return Pupil3D(center_left=None, center_ref=None, normal_left=None, normal_ref=None)

        X_left, _ = self.stereo.triangulate_best(eL.center, eR.center)

        X_ref = self.T_left_to_ref.apply_points(X_left)

        return Pupil3D(center_left=X_left, center_ref=X_ref, normal_left=None, normal_ref=None)

    def _in_front_cam(self, cam, X: np.ndarray) -> bool:
        """
        Preferred cheirality: check Z in the camera frame if R,t are available.
        Falls back to a projective-depth check using P if not.
        """
        if hasattr(cam, "R") and hasattr(cam, "t"):
            # Z_cam = (R X + t)_z
            Z = float(cam.R[2] @ X + cam.t[2])
            return Z > 0.0
        # Fallback to projective depth
        return self._in_front(cam.P, X)

    def _normalize_conic(self, Q: np.ndarray) -> np.ndarray:
        """Scale conic for numeric stability (Frobenius-norm = 1)."""
        Q = 0.5 * (Q + Q.T)
        s = float(np.linalg.norm(Q))
        return Q / (s + 1e-12)

    def _safe_solve_conic(self, Q: np.ndarray, l: np.ndarray) -> np.ndarray:
        """
        Solve Q x = l robustly. If Q is near-singular (tiny/degenerate ellipse),
        use light Tikhonov regularization tied to Q's scale.
        """
        try:
            return np.linalg.solve(Q, l)
        except np.linalg.LinAlgError:
            lam = 1e-9 * (np.abs(Q).trace() + 1e-12)
            return np.linalg.solve(Q + lam * np.eye(3), l)

    def _in_front(self, P: np.ndarray, X: np.ndarray) -> bool:
        """Cheirality check: point must project with positive depth."""
        Xh = np.array([X[0], X[1], X[2], 1.0], dtype=float)
        x = P @ Xh
        return x[2] > 0.0

    def _refine_plane_and_center(
            self,
            camL,
            camR,
            QL: np.ndarray,
            QR: np.ndarray,
            CLs: np.ndarray,
            CRs: np.ndarray,
            pi_init: np.ndarray,
            sL: float,
            sR: float,
            max_iters: int = 7,
            step0: float = 0.25,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Small projected-gradient refine over the pupil plane pi = [n; d] that
        minimizes:
            J(pi) = |pi^T CL* pi| + |pi^T CR* pi|
                    + w_in * (relu(sL * vL))^2
                    + w_in * (relu(sR * vR))^2
        where v? are conic values at the poles (the 'true' ellipse centers).
        Returns (pi_refined, pL_xy, pR_xy).
        """

        def plane_normalize(pi):
            n = pi[:3]
            norm = np.linalg.norm(n) + 1e-12
            return np.concatenate([n / norm, [pi[3] / norm]])

        def relu(t: float) -> float:
            return t if t > 0.0 else 0.0

        def pole_from_plane(P, Q, pi):
            # l = P * pi  (vanishing line), pole p s.t. Q p = l
            l = P @ pi
            p_h = self._safe_solve_conic(Q, l)
            if abs(p_h[2]) < 1e-12:
                return None, None
            x = float(p_h[0] / p_h[2])
            y = float(p_h[1] / p_h[2])
            xh = np.array([x, y, 1.0], dtype=float)
            v = float(xh @ (Q @ xh))  # signed conic value
            return (x, y), v

        def cost_and_poles(pi):
            rL = float(pi @ (CLs @ pi))
            rR = float(pi @ (CRs @ pi))
            pL, vL = pole_from_plane(camL.P, QL, pi)
            pR, vR = pole_from_plane(camR.P, QR, pi)
            if pL is None or pR is None:
                return np.inf, None, None
            w_in = 50.0
            penal = w_in * (relu(sL * vL) ** 2 + relu(sR * vR) ** 2)
            return abs(rL) + abs(rR) + penal, pL, pR

        pi = plane_normalize(pi_init.copy())
        step = step0

        bestJ, best_pL, best_pR = cost_and_poles(pi)
        if not np.isfinite(bestJ):
            return pi_init, (np.nan, np.nan), (np.nan, np.nan)

        for _ in range(max_iters):
            improved = False
            for k in range(4):
                for sign in (+1.0, -1.0):
                    pi_try = pi.copy()
                    pi_try[k] += sign * step
                    pi_try = plane_normalize(pi_try)
                    J, pL, pR = cost_and_poles(pi_try)
                    if J < bestJ:
                        bestJ, best_pL, best_pR = J, pL, pR
                        pi = pi_try
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break

        return pi, best_pL, best_pR

    def triangulate_center_using_cones(
            self,
            eL: Ellipse2D,  # left ellipse
            eR: Ellipse2D,  # right ellipse
    ) -> Pupil3D:
        """
        Perspective-correct 3D pupil center using dual view-cones, with
        a short refinement of the pupil plane and corrected image centers.
        Returns center + plane normals in LEFT and REF frames.
        """
        if eL is None or eR is None:
            return Pupil3D(center_left=None, center_ref=None, normal_left=None, normal_ref=None)

        def plane_normalize(pi: np.ndarray) -> np.ndarray:
            """Normalize homogeneous plane π = [n; d] so ||n|| = 1."""
            n = pi[:3]
            norm = np.linalg.norm(n) + 1e-12
            return np.concatenate([n / norm, [pi[3] / norm]])

        camL, camR = self.stereo.left, self.stereo.right

        # 1) image conics (normalized for numeric stability)
        QL = self._normalize_conic(ellipse_to_conic(eL))
        QR = self._normalize_conic(ellipse_to_conic(eR))

        # 2) lift to 3D view-cones: C = P^T Q P
        CL = build_view_cone(camL.P, QL)
        CR = build_view_cone(camR.P, QR)

        # 3) duals and candidate plane search via eigenvectors of scaled M
        CLs = adjugate_4x4(CL)
        CRs = adjugate_4x4(CR)
        M = CLs @ CLs + CRs @ CRs
        M = 0.5 * (M + M.T)
        M /= (np.linalg.norm(M) + 1e-12)
        evals, evecs = np.linalg.eigh(M)
        order = np.argsort(evals)
        # Try three smallest eigenvectors for more robustness (cheap)
        candidates = [evecs[:, order[i]] for i in range(min(3, len(order)))]

        # 4) determine inside/outside sign from the fitted ellipse centers
        sL = inside_sign(QL, eL.center)
        sR = inside_sign(QR, eR.center)

        best = None
        best_score = np.inf

        # 5) initial candidate scoring using poles and tangency residuals
        for pi in candidates:
            pi = plane_normalize(pi)
            lL = camL.P @ pi
            lR = camR.P @ pi

            pL_h = self._safe_solve_conic(QL, lL)
            pR_h = self._safe_solve_conic(QR, lR)
            if abs(pL_h[2]) < 1e-12 or abs(pR_h[2]) < 1e-12:
                continue
            pL = (float(pL_h[0] / pL_h[2]), float(pL_h[1] / pL_h[2]))
            pR = (float(pR_h[0] / pR_h[2]), float(pR_h[1] / pR_h[2]))

            okL = point_in_ellipse(QL, pL, sL)
            okR = point_in_ellipse(QR, pR, sR)
            rL = float(pi @ (CLs @ pi))
            rR = float(pi @ (CRs @ pi))
            score = abs(rL) + abs(rR) + (0.0 if (okL and okR) else 1e3)

            if score < best_score:
                best_score = score
                best = (pi, pL, pR, rL, rR)

        # Fallback if no plane worked: weak-perspective center triangulation
        if best is None:
            return self.triangulate_center(eL, eR)

        pi0, _pL0, _pR0, rL0, rR0 = best

        # 6) tiny refinement of plane and poles (early-exit if already great)
        # Use a relative tolerance based on the dual-cone scales
        tol = 1e-6 * (np.linalg.norm(CLs) + np.linalg.norm(CRs))
        max_iters_refine = 0 if abs(rL0) + abs(rR0) < tol else 5
        pi_ref, pL_ref, pR_ref = self._refine_plane_and_center(
            camL, camR, QL, QR, CLs, CRs, pi0, sL, sR,
            max_iters=max_iters_refine, step0=0.25
        )
        if not (np.isfinite(pL_ref[0]) and np.isfinite(pR_ref[0])):
            pL_ref, pR_ref, pi_ref = _pL0, _pR0, pi0
        else:
            pi_ref = plane_normalize(pi_ref)

        # 7) triangulate corrected centers; skip LM if already tight
        X_left, rms = self.stereo.triangulate_best(pL_ref, pR_ref, refine=False)
        if rms > 0.15:  # px; tune for your noise level
            X_left, _ = self.stereo.triangulate_best(
                pL_ref, pR_ref,
                refine=True,
                refine_thresh_px=0.15  # ensures threshold matches your check
            )

        # Cheirality guard
        if not (self._in_front_cam(camL, X_left) and self._in_front_cam(camR, X_left)):
            # --- helpers (local, no class state changed) ---
            def _recompute_poles_from_plane(P, Q, pi):
                # l = P @ pi  (vanishing line); pole p solves Q p = l
                l = P @ pi
                ph = self._safe_solve_conic(Q, l)  # robust solve with light Tikhonov
                if not np.isfinite(ph).all() or abs(ph[2]) < 1e-12:
                    return None
                return (float(ph[0] / ph[2]), float(ph[1] / ph[2]))

            # 1) Make plane normal face mean viewing direction (doesn't move plane)
            dL = camL.pixel_to_ray_world(*pL_ref)
            dR = camR.pixel_to_ray_world(*pR_ref)
            vmean = dL + dR
            if float(pi_ref[:3] @ vmean) < 0.0:
                pi_ref = -pi_ref  # flip orientation only

            # 2) Nudge plane *toward* cameras by shrinking |d|
            # (WORLD = LEFT, so cameras are near the origin; shrinking |d|
            # moves the plane closer to the cameras without changing its normal.)
            sL_sign = inside_sign(QL, eL.center)
            sR_sign = inside_sign(QR, eR.center)

            fixed = False
            for shrink in (0.9, 0.7, 0.5, 0.3):
                pi_try = pi_ref.copy()
                # preserve sign of d but reduce its magnitude
                pi_try[3] = np.sign(pi_try[3]) * shrink * abs(pi_try[3])

                pL_try = _recompute_poles_from_plane(camL.P, QL, pi_try)
                pR_try = _recompute_poles_from_plane(camR.P, QR, pi_try)
                if pL_try is None or pR_try is None:
                    continue

                # keep poles sensible: inside their ellipses
                if not (point_in_ellipse(QL, pL_try, sL_sign) and point_in_ellipse(QR, pR_try, sR_sign)):
                    continue

                X_try, rms_try = self.stereo.triangulate_best(pL_try, pR_try, refine=False)
                if self._in_front_cam(camL, X_try) and self._in_front_cam(camR, X_try):
                    # accept the nudge
                    pi_ref, pL_ref, pR_ref, X_left = pi_try, pL_try, pR_try, X_try
                    fixed = True
                    break

            # 3) Last resort: weak-perspective center triangulation
            if not fixed:
                return self.triangulate_center(eL, eR)

        # 8) plane normal (LEFT/world coords), unit length
        n_left = pi_ref[:3]
        n_left /= (np.linalg.norm(n_left) + 1e-12)

        # 9) map to requested reference frame
        X_ref = self.T_left_to_ref.apply_points(X_left)
        n_ref = self.T_left_to_ref.apply_dirs(n_left, normalize=True)

        return Pupil3D(
            center_left=X_left,
            center_ref=X_ref,
            normal_left=n_left,
            normal_ref=n_ref,
        )

    def find_pupil_center(self, img_left: np.ndarray, img_right: np.ndarray) -> Pupil3D:
        """
        Detect ellipses in L/R, then run the improved cone-based triangulation
        with plane + corrected-center refinement.
        """
        eL: Ellipse2D = self.ellipse_fitter.find_pupil_left(img_left)
        eR: Ellipse2D = self.ellipse_fitter.find_pupil_right(img_right)
        return self.triangulate_center_using_cones(eL, eR)



