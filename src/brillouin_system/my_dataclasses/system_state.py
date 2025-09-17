from dataclasses import dataclass

from brillouin_system.devices.cameras.andor.andor_dataclasses import AndorExposure, AndorCameraInfo
from brillouin_system.my_dataclasses.background_image import ImageStatistics


@dataclass
class SystemState:
    is_reference_mode: bool
    is_do_bg_subtraction_active: bool
    andor_camera_info: AndorCameraInfo
    bg_image: ImageStatistics | None
    dark_image: ImageStatistics | None


