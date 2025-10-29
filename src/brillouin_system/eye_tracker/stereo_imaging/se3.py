
"""
coordinates.py — minimal SE(3) frames and coordinate conversion utilities (Python 3.10+)

Keep named frames (e.g., "left", "right", "zaber") and convert points or directions
between them. Includes helpers to estimate rigid transforms from 3D–3D correspondences
and to build a frame from measured stage axes.

Dependencies: numpy
Python: 3.10+
"""

from __future__ import annotations  # harmless on 3.10+, enables forward refs without quotes

import json
from dataclasses import dataclass
import numpy as np


# -----------------------------
# Core SE(3) representation
# -----------------------------

def _as_Rt(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

def save_se3_json(T: SE3, path: str | bytes) -> None:
    """
    Save an SE3 transform to a JSON file.

    Parameters
    ----------
    T : SE3
        The transform to save.
    path : str or Path
        Path to the JSON file.
    """
    data = {
        "R": T.R.tolist(),
        "t": T.t.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_se3_json(path: str | bytes) -> SE3:
    """
    Load an SE3 transform from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    SE3
        The loaded transform.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    R = np.array(data["R"], dtype=float)
    t = np.array(data["t"], dtype=float)
    return SE3(R, t)


