from __future__ import annotations
import json
from dataclasses import dataclass
from os import PathLike
import numpy as np
from typing import Union, Tuple

def _as_Rt(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    return R, t

@dataclass(frozen=True)
class SE3:
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        R, t = _as_Rt(self.R, self.t)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    def inv(self) -> "SE3":
        Rinv = self.R.T
        return SE3(Rinv, -Rinv @ self.t)

    def __matmul__(self, other: "SE3") -> "SE3":
        R1, t1 = _as_Rt(self.R, self.t)
        R2, t2 = _as_Rt(other.R, other.t)
        return SE3(R1 @ R2, R1 @ t2 + t1)

    def apply_points(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return self.R @ X + self.t
        if X.ndim == 2 and X.shape[1] == 3:
            return (self.R @ X.T).T + self.t
        raise ValueError("X must be shape (3,) or (N,3)")

    def apply_dirs(self, D: np.ndarray, normalize: bool = False) -> np.ndarray:
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

def save_se3_json(T: SE3, path: Union[str, PathLike[str]]) -> None:
    data = {"R": T.R.tolist(), "t": T.t.tolist()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_se3_json(path: Union[str, PathLike[str]]) -> SE3:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # be tolerant if extra keys exist
    R = np.asarray(data["R"], dtype=float).reshape(3, 3)
    t = np.asarray(data["t"], dtype=float).reshape(3)
    return SE3(R, t)
