from dataclasses import dataclass

import numpy as np

from brillouin_system.devices.cameras.allied.base_dual_cameras import BaseDualCameras

@dataclass
class EyeTrackingResults:
    left_frame: np.ndarray
    right_frame: np.ndarray
    pupil_center: tuple


class EyeTracker:

    def __init__(self,
                 dual_cameras: BaseDualCameras):
        self.dual_cameras: BaseDualCameras = dual_cameras



