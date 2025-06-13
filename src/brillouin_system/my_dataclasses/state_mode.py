from dataclasses import dataclass

from brillouin_system.my_dataclasses.background_image import ImageStatistics
from brillouin_system.my_dataclasses.camera_settings import AndorCameraSettings


@dataclass
class StateMode:
    is_reference_mode: bool
    is_do_bg_subtraction_active: bool
    camera_settings: AndorCameraSettings
    bg_image: ImageStatistics | None
    dark_image: ImageStatistics | None


