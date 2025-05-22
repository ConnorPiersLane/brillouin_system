from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.camera_settings import CameraSettings


@dataclass
class BackgroundData:
    image: np.ndarray
    camera_settings: CameraSettings