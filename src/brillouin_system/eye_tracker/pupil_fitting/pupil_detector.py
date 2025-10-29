# pupil_detector.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D
from brillouin_system.eye_tracker.pupil_fitting.helpers import ellipse_to_conic, build_view_cone, \
    adjugate_4x4, inside_sign, point_in_ellipse
from brillouin_system.eye_tracker.stereo_imaging.se3 import SE3
from brillouin_system.eye_tracker.stereo_imaging.stereo_cameras import StereoCameras
from ellipse_fitter import EllipseFitter  # your class & type


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

        uvL = eL.center
        uvR = eR.center

        # Prefer the rig's refine-capable triangulator; falls back to midpoint if absent.

        X_left, _rms = self.stereo.triangulate_best(uvL, uvR, refine=True)

        X_ref = self.T_left_to_ref.apply_points(X_left)

        return Pupil3D(center_left=X_left, center_ref=X_ref, normal_left=None, normal_ref=None)


    def triangulate_center_using_cones(
        self,
        eL: Ellipse2D,  # left ellipse
        eR: Ellipse2D,  # right ellipse
    ) -> Pupil3D:
        """
        Perspective-correct 3D pupil center using dual view-cones.
        Also returns the pupil plane normal (LEFT/world) and its mapped REF version.
        """
        if eL is None or eR is None:
            return Pupil3D(center_left=None, center_ref=None, normal_left=None, normal_ref=None)

        camL, camR = self.stereo.left, self.stereo.right

        # 1) image conics
        QL = ellipse_to_conic(eL)
        QR = ellipse_to_conic(eR)

        # 2) lift to 3D view-cones: C = P^T Q P
        CL = build_view_cone(camL.P, QL)
        CR = build_view_cone(camR.P, QR)

        # 3) duals (adjugates) and candidate plane search
        CLs = adjugate_4x4(CL)
        CRs = adjugate_4x4(CR)
        M = CLs @ CLs + CRs @ CRs
        M = 0.5 * (M + M.T)
        evals, evecs = np.linalg.eigh(M)
        order = np.argsort(evals)
        candidates = [evecs[:, order[i]] for i in range(min(3, len(order)))]

        sL = inside_sign(QL, eL.center)
        sR = inside_sign(QR, eR.center)

        best = None
        best_score = np.inf

        for pi in candidates:
            # normalize plane vector (unit normal part) for stability
            pi = pi / (np.linalg.norm(pi[:3]) + 1e-12)

            # 4) vanishing lines and true projected centers
            lL = camL.P @ pi
            lR = camR.P @ pi
            try:
                pL_h = np.linalg.solve(QL, lL)
                pR_h = np.linalg.solve(QR, lR)
            except np.linalg.LinAlgError:
                continue
            if abs(pL_h[2]) < 1e-12 or abs(pR_h[2]) < 1e-12:
                continue
            pL = (float(pL_h[0] / pL_h[2]), float(pL_h[1] / pL_h[2]))
            pR = (float(pR_h[0] / pR_h[2]), float(pR_h[1] / pR_h[2]))

            # inside checks + tangency residuals
            okL = point_in_ellipse(QL, pL, sL)
            okR = point_in_ellipse(QR, pR, sR)
            rL = float(pi @ (CLs @ pi))
            rR = float(pi @ (CRs @ pi))
            score = abs(rL) + abs(rR) + (0.0 if (okL and okR) else 1e3)

            if score < best_score:
                best_score = score
                best = (pi, pL, pR)

        # fallback if no plane worked: center triangulation
        if best is None:
            return self.triangulate_center(eL, eR)

        pi, pL, pR = best

        # 5) triangulate corrected centers with the rig
        X_left, _rms = self.stereo.triangulate_best(pL, pR, refine=True)


        # plane normal (LEFT/world coords), unit length
        n_left = pi[:3]
        n_left /= (np.linalg.norm(n_left) + 1e-12)

        # map to reference frame
        X_ref = self.T_left_to_ref.apply_points(X_left)
        n_ref = self.T_left_to_ref.apply_dirs(n_left, normalize=True)

        return Pupil3D(center_left=X_left, center_ref=X_ref,
                       normal_left=n_left, normal_ref=n_ref)
