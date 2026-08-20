from dataclasses import dataclass

from brillouin_system.devices.cameras.andor.andor_dataclasses import AndorCameraInfo
from brillouin_system.my_dataclasses.background_image import ImageStatistics


@dataclass
class SystemState:
    # The background-frame subtraction feature (is_do_bg_subtraction_active /
    # bg_image) was removed 2026-08-20: never used in production, not planned.
    # Old files carrying those fields still load; the values are dropped.
    is_reference_mode: bool
    andor_camera_info: AndorCameraInfo
    dark_image: ImageStatistics | None
