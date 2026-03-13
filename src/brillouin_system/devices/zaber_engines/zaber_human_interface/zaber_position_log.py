from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ZaberPositionLog:
    t_perf: np.ndarray      # shape (n,), float64
    z_um: np.ndarray        # shape (n,), float64


def interp_z_positions(t_query: float, t_z: np.ndarray, z_um: np.ndarray) -> float:
    """Linear interpolation with monotonic-time cleanup (drop non-increasing timestamps)."""
    t_z = np.asarray(t_z, dtype=np.float64)
    z_um = np.asarray(z_um, dtype=np.float64)
    if t_z.size < 2:
        return float("nan")
    keep = np.concatenate(([True], np.diff(t_z) > 0))
    t_z = t_z[keep]
    z_um = z_um[keep]
    if t_z.size < 2:
        return float("nan")
    return float(np.interp(np.array([t_query], dtype=np.float64), t_z, z_um)[0])