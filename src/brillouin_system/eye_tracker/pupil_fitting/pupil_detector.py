# pupil_detector.py
from __future__ import annotations
import numpy as np

from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D
from brillouin_system.eye_tracker.pupil_fitting.pupil3D import Pupil3D
from brillouin_system.eye_tracker.pupil_fitting.pupil_detector_helpers import ellipse_to_conic, build_view_cone, \
    adjugate_4x4, inside_sign, point_in_ellipse, _image_line_from_plane
from brillouin_system.eye_tracker.stereo_imaging.se3 import SE3
from brillouin_system.eye_tracker.stereo_imaging.stereo_cameras import StereoCameras




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
        stereo_cameras: StereoCameras,                      # type:
        left_to_ref: SE3 = None,   # LEFT -> REF; if None, identity (left)
    ) -> None:
        self.stereo = stereo_cameras

        if left_to_ref is None:
            self.T_left_to_ref = SE3(np.eye(3), np.zeros(3))
        else:
            self.T_left_to_ref = left_to_ref

    # ---------------- convenience ----------------
    def set_reference(self, T_left_to_ref: SE3) -> None:
        """Update the output reference frame (LEFT -> REF)."""
        self.T_left_to_ref = T_left_to_ref


    def triangulate_center(
        self,
        eL: Ellipse2D,
        eR: Ellipse2D,
    ) -> Pupil3D | None:
        """
        Simple baseline: triangulate the image-ellipse centers (weak perspective).
        Uses the rig's robust triangulator and then maps to the selected reference via SE3.
        """
        if eL is None or eR is None:
            return None

        X_left, _ = self.stereo.triangulate_best(eL.center, eR.center)

        X_ref = self.T_left_to_ref.apply_points(X_left)

        return Pupil3D(center_left=X_left, center_ref=X_ref, normal_left=None, normal_ref=None, radius=None)

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

    def _safe_solve_conic(self, Q: np.ndarray, l: np.ndarray) -> np.ndarray | None:
        """
        Solve Q x = l robustly. If Q is near-singular (tiny/degenerate ellipse),
        use light Tikhonov regularization tied to Q's Frobenius norm.

        Returns:
            ph : (3,) homogeneous solution, or None if Q is too ill-conditioned.
        """
        Q = 0.5 * (Q + Q.T)
        normQ = float(np.linalg.norm(Q))
        if normQ < 1e-12:
            return None

        # Try direct solve
        try:
            ph = np.linalg.solve(Q, l)
        except np.linalg.LinAlgError:
            ph = None

        # If direct solve fails, use Tikhonov-regularized solves
        if ph is None:
            lam_base = 1e-9 * (normQ + 1e-12)
            for scale in (1.0, 1e3):
                lam = lam_base * scale
                try:
                    ph = np.linalg.solve(Q + lam * np.eye(3), l)
                    break
                except np.linalg.LinAlgError:
                    ph = None

            if ph is None:
                return None

        # Reject extremely ill-conditioned Q (degenerate ellipse)
        svals = np.linalg.svd(Q, compute_uv=False)
        if svals[-1] < 1e-15 or (svals[0] / max(svals[-1], 1e-15)) > 1e12:
            return None

        return ph


    def _in_front(self, P: np.ndarray, X: np.ndarray) -> bool:
        """Cheirality check: point must project with positive depth."""
        Xh = np.array([X[0], X[1], X[2], 1.0], dtype=float)
        x = P @ Xh
        return x[2] > 0.0

    def _radius_from_cone_and_plane(self, C: np.ndarray, X0: np.ndarray, n_hat: np.ndarray) -> float | None:
        """
        Recover metric circle radius by intersecting a (view) cone C (4x4, primal quadric)
        with the pupil plane through X0 with unit normal n_hat. Uses a plane-local
        orthonormal basis and reads radius from the induced 2D conic.
        """

        # 1) Build orthonormal basis {U, V} for the plane (||U||=||V||=1, U ⟂ V ⟂ n)
        n = n_hat / (np.linalg.norm(n_hat) + 1e-12)
        # Pick a vector not parallel to n
        a = np.array([1.0, 0.0, 0.0])
        if abs(n @ a) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        U = np.cross(n, a)
        U /= (np.linalg.norm(U) + 1e-12)
        V = np.cross(n, U)
        V /= (np.linalg.norm(V) + 1e-12)

        # 2) Map (u,v,1) -> [X0 + u U + v V; 1]
        B = np.array([
            [U[0], V[0], X0[0]],
            [U[1], V[1], X0[1]],
            [U[2], V[2], X0[2]],
            [0.0, 0.0, 1.0],
        ], dtype=float)  # 4x3

        # 3) Induced plane conic Qp (3x3) in (u,v,1)
        Qp = B.T @ C @ B
        Qp = 0.5 * (Qp + Qp.T)

        # 4) (Tiny) recenter so linear term ~ 0: u^T A u + 2 b^T u + c = 0
        A = Qp[:2, :2]
        b = Qp[:2, 2]
        c = Qp[2, 2]
        # If A is ill-conditioned, bail out gracefully
        try:
            u0 = -np.linalg.solve(A, b)  # conic center in (u,v)
        except np.linalg.LinAlgError:
            return None

        if not np.isfinite(u0).all():
            return None

        # Translate to the center: T = [[1,0,-u0x],[0,1,-u0y],[0,0,1]]
        T = np.array([[1.0, 0.0, -u0[0]],
                      [0.0, 1.0, -u0[1]],
                      [0.0, 0.0, 1.0]], dtype=float)
        Qc = T.T @ Qp @ T
        Qc = 0.5 * (Qc + Qc.T)

        A2 = Qc[:2, :2]
        c2 = Qc[2, 2]

        # Ensure we really have an ellipse on the plane
        wA2, _ = np.linalg.eigh(A2)
        lam_min = float(wA2.min())
        lam_max = float(wA2.max())
        if lam_min <= 1e-12:
            return None

        # Reject very elongated intersections (very non-circular)
        condA2 = lam_max / max(lam_min, 1e-12)
        if condA2 > 1e3:
            return None

        # Average eigenvalues for robustness (circle ⇒ A2 ∝ I)
        alpha = 0.5 * (lam_min + lam_max)
        if alpha <= 0 or -c2 <= 0:
            return None


        r = float(np.sqrt(max(1e-12, (-c2) / max(alpha, 1e-12))))
        return r if np.isfinite(r) and r > 0 else None

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
            l = _image_line_from_plane(P, pi)
            p_h = self._safe_solve_conic(Q, l)
            if p_h is None or not np.isfinite(p_h).all() or abs(p_h[2]) < 1e-12:
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
            return Pupil3D(center_left=None, center_ref=None, normal_left=None, normal_ref=None, radius=None)

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

        # Use a linear combination; avoids over-emphasizing large entries
        M = CLs + CRs
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
            lL = _image_line_from_plane(camL.P, pi)
            lR = _image_line_from_plane(camR.P, pi)

            pL_h = self._safe_solve_conic(QL, lL)
            pR_h = self._safe_solve_conic(QR, lR)
            if (
                pL_h is None or pR_h is None
                or not np.isfinite(pL_h).all()
                or not np.isfinite(pR_h).all()
                or abs(pL_h[2]) < 1e-12
                or abs(pR_h[2]) < 1e-12
            ):
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
        tol = 1e-6 * (
                np.linalg.norm(CLs) + np.linalg.norm(CRs) +
                np.linalg.norm(QL) + np.linalg.norm(QR)
        )

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

        # 7b) Cheirality guard
        if not (self._in_front_cam(camL, X_left) and self._in_front_cam(camR, X_left)):
            # --- helpers (local, no class state changed) ---
            def _recompute_poles_from_plane(P, Q, pi):
                # l = P @ pi  (vanishing line); pole p solves Q p = l
                l = _image_line_from_plane(P, pi)
                ph = self._safe_solve_conic(Q, l)  # robust solve with light Tikhonov
                if ph is None or not np.isfinite(ph).all() or abs(ph[2]) < 1e-12:
                    return None
                return (float(ph[0] / ph[2]), float(ph[1] / ph[2]))

            # Sign for "inside" classification
            sL_sign = inside_sign(QL, eL.center)
            sR_sign = inside_sign(QR, eR.center)

            # 1) Make plane normal face mean viewing direction (doesn't move plane)
            dL = camL.pixel_to_ray_world(*pL_ref)
            dR = camR.pixel_to_ray_world(*pR_ref)
            vmean = dL + dR
            if float(pi_ref[:3] @ vmean) < 0.0:
                pi_ref = -pi_ref  # flip orientation only

                # After flipping, recompute poles so they are consistent with the new plane
                pL_new = _recompute_poles_from_plane(camL.P, QL, pi_ref)
                pR_new = _recompute_poles_from_plane(camR.P, QR, pi_ref)
                if (
                    pL_new is not None and pR_new is not None
                    and point_in_ellipse(QL, pL_new, sL_sign)
                    and point_in_ellipse(QR, pR_new, sR_sign)
                ):
                    pL_ref, pR_ref = pL_new, pR_new

            # 2) Nudge plane *toward* cameras by shrinking |d|
            # (WORLD = LEFT, so cameras are near the origin; shrinking |d|
            # moves the plane closer to the cameras without changing its normal.)
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
                if not (
                    point_in_ellipse(QL, pL_try, sL_sign)
                    and point_in_ellipse(QR, pR_try, sR_sign)
                ):
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

        # 9) After you compute X_left (center) and n_left (unit normal) and before mapping to REF:
        # --- Robust radius from each cone ---
        radiusL = self._radius_from_cone_and_plane(CL, X_left, n_left)
        radiusR = self._radius_from_cone_and_plane(CR, X_left, n_left)

        # --- Weight by how front-on each camera is: w ~ cos^2(theta) ---
        # Use the corrected centers you already computed.
        dL = camL.pixel_to_ray_world(*pL_ref)
        dL = dL / (np.linalg.norm(dL) + 1e-12)
        dR = camR.pixel_to_ray_world(*pR_ref)
        dR = dR / (np.linalg.norm(dR) + 1e-12)

        cL = abs(float(n_left @ dL))
        cR = abs(float(n_left @ dR))
        wL = cL * cL
        wR = cR * cR

        # --- Combine robustly (handle None cases) ---
        if (radiusL is not None) and (radiusR is not None):
            if (wL + wR) > 1e-12:
                radius = (wL * radiusL + wR * radiusR) / (wL + wR)
            else:
                radius = 0.5 * (radiusL + radiusR)  # degenerate view; fall back
        elif radiusL is not None:
            radius = radiusL
        else:
            radius = radiusR


        # 10) map to requested reference frame
        X_ref = self.T_left_to_ref.apply_points(X_left)
        n_ref = self.T_left_to_ref.apply_dirs(n_left, normalize=True)

        return Pupil3D(
            center_left=X_left,
            center_ref=X_ref,
            normal_left=n_left,
            normal_ref=n_ref,
            radius=radius,
        )




