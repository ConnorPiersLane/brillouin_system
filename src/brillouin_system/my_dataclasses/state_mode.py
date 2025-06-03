from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.camera_settings import CameraSettings


@dataclass
class StateMode:
    is_do_bg_subtraction_active: bool
    bg_image: np.ndarray | None
    dark_noise_image: np.ndarray | None
    camera_settings: CameraSettings

