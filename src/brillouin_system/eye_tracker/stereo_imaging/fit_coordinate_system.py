import numpy as np

from brillouin_system.eye_tracker.stereo_imaging.coord_transformer import SE3


# --- core solver (Umeyama/Kabsch), with optional scale ---
def _umeyama(A: np.ndarray, B: np.ndarray, *, with_scale: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find R, t, s minimizing sum || s*R*A_i + t - B_i ||^2
    A: (N,3) in left,  B: (N,3) in zaber
    Returns (R, t, s)
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("A and B must both be shape (N,3)")
    N = A.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 non-collinear points")

    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    Ac = A - muA
    Bc = B - muB

    H = Ac.T @ Bc / N
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Enforce right-handed rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if with_scale:
        varA = (Ac**2).sum() / N
        s = (S.sum()) / (varA + 1e-15)
    else:
        s = 1.0

    t = muB - s * (R @ muA)
    return R, t, float(s)

def _residuals(R: np.ndarray, t: np.ndarray, s: float, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    pred = (s * (R @ A.T)).T + t  # (N,3)
    return np.linalg.norm(pred - B, axis=1)

def fit_coordinate_system(
    points_left: np.ndarray,
    points_zaber: np.ndarray,
    *,
    with_scale: bool = False,
    trim_fraction: float = 0.0,
    trim_repeats: int = 1
) -> tuple[SE3, dict]:
    """
    Fit T_left_to_zaber from 3D-3D correspondences.

    Parameters
    ----------
    points_left : (N,3)
        Measured points in LEFT.
    points_zaber : (N,3)
        Corresponding points in ZABER.
    with_scale : bool
        If True, also estimate a global scale s (useful if LEFT and ZABER are in different length units).
        If False (default), fit a rigid transform (s = 1).
    trim_fraction : float in [0, 0.5)
        Robust trimming fraction. E.g., 0.2 keeps the best 80% inliers and refits.
        Set 0.0 to disable trimming.
    trim_repeats : int
        How many trim-refit cycles to run.

    Returns
    -------
    (SE3, info)
      SE3: T_left_to_zaber such that X_zaber ≈ R @ X_left + t (times s if with_scale)
      info: dict with fields:
        - 'scale': float
        - 'rms': float (RMS error on final inliers, in ZABER units)
        - 'residuals': per-point residuals (on final inliers ordering)
        - 'inlier_mask': boolean mask over the original inputs (True = kept)
        - 'num_inliers': int
    """
    A = np.asarray(points_left, float)
    B = np.asarray(points_zaber, float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("points_left and points_zaber must both be (N,3)")
    N = A.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 non-collinear correspondences")

    # Initial fit
    R, t, s = _umeyama(A, B, with_scale=with_scale)
    res = _residuals(R, t, s, A, B)
    inlier_mask = np.ones(N, dtype=bool)

    # Optional robust trimming
    trim_fraction = float(trim_fraction)
    if not (0.0 <= trim_fraction < 0.5):
        raise ValueError("trim_fraction must be in [0.0, 0.5)")
    for _ in range(trim_repeats if trim_fraction > 0 else 0):
        keep = int(np.ceil((1.0 - trim_fraction) * inlier_mask.sum()))
        order = np.argsort(res)
        new_mask = np.zeros_like(inlier_mask)
        new_mask[order[:keep]] = True
        if new_mask.sum() < 3:
            break
        if np.all(new_mask == inlier_mask):
            break
        inlier_mask = new_mask
        R, t, s = _umeyama(A[inlier_mask], B[inlier_mask], with_scale=with_scale)
        res = _residuals(R, t, s, A, B)

    # Final diagnostics (on inliers)
    res_in = res[inlier_mask]
    rms = float(np.sqrt(np.mean(res_in**2))) if res_in.size else float("nan")

    # Package SE3 (store only rigid part). If with_scale=True, caller can keep 'scale'.
    T_left_to_zaber = SE3(R=R, t=t)

    info = {
        "scale": s,
        "rms": rms,
        "residuals": res_in,
        "inlier_mask": inlier_mask,
        "num_inliers": int(inlier_mask.sum()),
    }
    return T_left_to_zaber, info
