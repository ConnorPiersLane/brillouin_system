# ----------------------------------------------------------------------
# Conic & plane utilities (for perspective-correct centers)
# ----------------------------------------------------------------------
import numpy as np

from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D


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

def inside_sign(Q: np.ndarray, center_uv: tuple[float,float]) -> float:
    """
    Determine the sign convention for 'inside' of an ellipse conic Q by evaluating at its center.
    """
    u, v = center_uv
    x = np.array([u, v, 1.0], dtype=np.float64)
    val = float(x @ Q @ x)
    # Inside should be opposite sign of boundary (0). Use sign at center as inside.
    return -1.0 if val > 0 else 1.0

def point_in_ellipse(Q: np.ndarray, uv: tuple[float,float], sign_inside: float) -> bool:
    u, v = uv
    x = np.array([u, v, 1.0], dtype=np.float64)
    return sign_inside * float(x @ Q @ x) < 0.0
