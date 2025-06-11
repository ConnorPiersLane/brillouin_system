from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.background_image import ImageStatistics
from brillouin_system.my_dataclasses.camera_settings import CameraSettings


@dataclass
class StateMode:
    is_reference_mode: bool
    is_do_bg_subtraction_active: bool
    bg_image: ImageStatistics | None
    dark_image: ImageStatistics | None
    camera_settings: CameraSettings

