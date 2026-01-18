from dataclasses import dataclass

import numpy as np


@dataclass
class Pupil3D:
    """Container for 3D pupil results."""
    center_left: np.ndarray | None   # (3,) in LEFT camera frame
    center_ref:  np.ndarray | None  # (3,) in reference frame (via SE3)
    normal_left: np.ndarray | None = None   # (3,) optional pupil normal in LEFT
    normal_ref:  np.ndarray | None = None   # (3,) optional pupil normal in REF
    radius: float | None = None