
"""
coordinates.py — minimal SE(3) frames and coordinate conversion utilities (Python 3.10+)

Keep named frames (e.g., "left", "right", "zaber") and convert points or directions
between them. Includes helpers to estimate rigid transforms from 3D–3D correspondences
and to build a frame from measured stage axes.

Dependencies: numpy
Python: 3.10+
"""

from __future__ import annotations  # harmless on 3.10+, enables forward refs without quotes
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import numpy as np


# -----------------------------
# Core SE(3) representation
# -----------------------------

def _as_Rt(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    return R, t


@dataclass(frozen=True)
class SE3:
    """A simple rigid transform: x' = R x + t.

    Attributes
    ----------
    R : (3,3) rotation matrix
    t : (3,) translation vector
    """

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        R, t = _as_Rt(self.R, self.t)
        # validate a bit
        if R.shape != (3, 3):
            raise ValueError("R must be 3x3")
        if t.shape != (3,):
            raise ValueError("t must be (3,)")
        # freeze normalized copies
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    def inv(self) -> SE3:
        """Inverse transform."""
        Rinv = self.R.T
        return SE3(Rinv, -Rinv @ self.t)

    def __matmul__(self, other: SE3) -> SE3:
        """Compose two transforms: self ∘ other."""
        R1, t1 = _as_Rt(self.R, self.t)
        R2, t2 = _as_Rt(other.R, other.t)
        return SE3(R1 @ R2, R1 @ t2 + t1)

    # --- apply ---
    def apply_points(self, X: np.ndarray) -> np.ndarray:
        """Apply to points of shape (3,) or (N,3)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return self.R @ X + self.t
        elif X.ndim == 2 and X.shape[1] == 3:
            return (self.R @ X.T).T + self.t
        else:
            raise ValueError("X must be shape (3,) or (N,3)")

    def apply_dirs(self, D: np.ndarray, normalize: bool = False) -> np.ndarray:
        """Apply to direction vectors (no translation).

        If normalize=True, re-normalize output directions to unit length.
        """
        D = np.asarray(D, dtype=float)
        if D.ndim == 1:
            out = self.R @ D
        elif D.ndim == 2 and D.shape[1] == 3:
            out = (self.R @ D.T).T
        else:
            raise ValueError("D must be shape (3,) or (N,3)")
        if normalize:
            if out.ndim == 1:
                n = np.linalg.norm(out) + 1e-15
                out = out / n
            else:
                n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-15
                out = out / n
        return out


# ----------------------------------------
# Frame graph & coordinate transformer
# ----------------------------------------

class CoordTransformer:
    """Registry of named frames and SE(3) edges with search + caching.

    Example
    -------
    xf = CoordTransformer(hub="left")   # treat LEFT as the global reference
    xf.set("left", "zaber", T_left_to_zaber)
    Pz = xf.points(P_left, "left", "zaber")
    """
    def __init__(self, hub: str | None = None):
        self.edges: Dict[tuple[str, str], SE3] = {}
        self.cache: Dict[tuple[str, str], SE3] = {}
        self.hub: str | None = hub

    # ----- registration -----
    def set(self, src: str, dst: str, T_src_to_dst: SE3) -> None:
        """Register a transform and its inverse."""
        self.edges[(src, dst)] = T_src_to_dst
        self.edges[(dst, src)] = T_src_to_dst.inv()
        # purge cache entries that involve src or dst
        to_del = [k for k in self.cache if src in k or dst in k]
        for k in to_del:
            del self.cache[k]

    def has(self, src: str, dst: str) -> bool:
        return (src, dst) in self.edges

    # ----- path finding -----
    def _find_path_transform(self, src: str, dst: str) -> SE3:
        if src == dst:
            return SE3(np.eye(3), np.zeros(3))

        key = (src, dst)
        if key in self.cache:
            return self.cache[key]

        # hub fast-path (e.g., going to or from "left")
        if (self.hub is not None) and (src == self.hub or dst == self.hub):
            try:
                T = self.edges[(src, dst)]
                self.cache[key] = T
                return T
            except KeyError:
                pass  # fall back to BFS

        # Build adjacency list
        from collections import defaultdict, deque
        graph = defaultdict(list)
        for (a, b), T in self.edges.items():
            graph[a].append(b)

        # BFS over small graph is perfectly adequate
        q = deque([src])
        parent: Dict[str, str | None] = {src: None}

        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in graph[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)

        if dst not in parent:
            raise ValueError(f"No transform path from '{src}' to '{dst}'")

        # Reconstruct path (backtrack) and compose
        nodes: List[str] = []
        cur: str | None = dst
        while cur is not None:
            nodes.append(cur)
            cur = parent[cur]
        nodes = nodes[::-1]  # src ... dst

        T_total = SE3(np.eye(3), np.zeros(3))
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i+1]
            T_total = self.edges[(a, b)] @ T_total

        self.cache[key] = T_total
        return T_total

    # ----- public API -----
    def transform(self, X: np.ndarray, src: str, dst: str, *, is_direction: bool = False,
                  normalize_dirs: bool = False) -> np.ndarray:
        """Transform points or directions from frame `src` to `dst`.

        Parameters
        ----------
        X : array-like
            (3,) or (N,3) points/directions in the `src` frame.
        src, dst : str
            Source and destination frame names.
        is_direction : bool
            If True, interpret X as directions (no translation).
        normalize_dirs : bool
            If True and is_direction, re-normalize output directions.
        """
        T = self._find_path_transform(src, dst)
        if is_direction:
            return T.apply_dirs(X, normalize=normalize_dirs)
        else:
            return T.apply_points(X)

    # sugar
    def points(self, X: np.ndarray, src: str, dst: str) -> np.ndarray:
        return self.transform(X, src, dst, is_direction=False)

    def dirs(self, D: np.ndarray, src: str, dst: str, *, normalize: bool=False) -> np.ndarray:
        return self.transform(D, src, dst, is_direction=True, normalize_dirs=normalize)


# ---------------------------------------------------
# Helpers: estimate SE(3) from 3D-3D correspondences
# ---------------------------------------------------

def estimate_se3_umeyama(A_src: np.ndarray, B_dst: np.ndarray, allow_reflection: bool=False) -> SE3:
    """Estimate rigid transform T such that B ≈ R A + t (A->B).

    Parameters
    ----------
    A_src : (N,3) array
        Points in source frame A (e.g., left).
    B_dst : (N,3) array
        Corresponding points in destination frame B (e.g., zaber).
    allow_reflection : bool
        If True, allow a reflection if necessary. Default keeps right-handedness.

    Returns
    -------
    SE3
        Transform mapping A->B.
    """
    A = np.asarray(A_src, dtype=float)
    B = np.asarray(B_dst, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("A and B must both be (N,3)")

    # Center
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)

    # Covariance and SVD
    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        # enforce proper rotation (right-handed) unless reflection explicitly allowed
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = B.mean(axis=0) - R @ A.mean(axis=0)
    return SE3(R, t)


# -------------------------------------------------
# Helpers: build a frame from two measured axes
# -------------------------------------------------

def orthonormal_from_two(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Build an orthonormal basis from two non-collinear vectors.

    Returns a 3x3 rotation whose columns are (ex, ey, ez).
    """
    vx = np.asarray(vx, dtype=float).reshape(3)
    vy = np.asarray(vy, dtype=float).reshape(3)
    ex = vx / (np.linalg.norm(vx) + 1e-15)
    vy_proj = vy - (vy @ ex) * ex
    ey = vy_proj / (np.linalg.norm(vy_proj) + 1e-15)
    ez = np.cross(ex, ey)
    # ensure right-handed and orthonormal
    ez /= (np.linalg.norm(ez) + 1e-15)
    ey = np.cross(ez, ex)
    return np.column_stack([ex, ey, ez])  # columns

def frame_from_axes(origin: np.ndarray, x_dir: np.ndarray, y_dir: np.ndarray, *, origin_is_in_left: bool=True) -> SE3:
    """Create SE3 for a frame defined by origin and two axis directions.

    If origin_is_in_left=True, returns T_left_to_frame (maps left -> frame).
    If False, returns T_frame_to_left.
    """
    O = np.asarray(origin, dtype=float).reshape(3)
    R_cols = orthonormal_from_two(np.asarray(x_dir), np.asarray(y_dir))  # columns ex ey ez
    R_l_to_f = R_cols.T  # change of basis: coordinates in new frame
    if origin_is_in_left:
        t = -R_l_to_f @ O
        return SE3(R_l_to_f, t)  # left -> frame
    else:
        # Build frame->left
        return SE3(R_cols, O)


# ----------------------
# Small numeric helpers
# ----------------------

def is_rotation_matrix(R: np.ndarray, atol: float = 1e-6) -> bool:
    R = np.asarray(R, float).reshape(3, 3)
    should_be_I = R.T @ R
    I = np.eye(3)
    return np.allclose(should_be_I, I, atol=atol) and np.isclose(np.linalg.det(R), 1.0, atol=atol)


# ----------------------
# Example (remove if needed)
# ----------------------

if __name__ == "__main__":
    # Quick self-test: compose and invert
    T1 = SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))
    ang = np.deg2rad(30.0)
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                   [np.sin(ang),  np.cos(ang), 0.0],
                   [0.0,          0.0,         1.0]])
    T2 = SE3(Rz, np.array([0.0, 2.0, 0.0]))
    T = T2 @ T1

    Xl = np.array([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
    Xp = T.apply_points(Xl)
    Xl_back = T.inv().apply_points(Xp)
    assert np.allclose(Xl, Xl_back, atol=1e-9)

    xf = CoordTransformer(hub="left")
    xf.set("left", "zaber", SE3(np.eye(3), np.array([10.0, 0.0, 0.0])))
    Pl = np.array([0.1, 0.2, 0.3])
    Pz = xf.points(Pl, "left", "zaber")
    Pl2 = xf.points(Pz, "zaber", "left")
    assert np.allclose(Pl, Pl2, atol=1e-12)

    print("Self-test OK (3.10+)")
