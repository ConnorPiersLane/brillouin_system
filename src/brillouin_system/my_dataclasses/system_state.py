from dataclasses import dataclass

from brillouin_system.devices.cameras.andor.andor_dataclasses import AndorCameraInfo


@dataclass
class SystemState:
    # Removed fields, dropped on load from old files (h5 drops unknown
    # fields, pickles keep them as inert attributes):
    #   is_do_bg_subtraction_active / bg_image  (bg-frame subtraction,
    #     removed 2026-08-20 — never used in production)
    #   dark_image  (dark-frame capture, removed 2026-08-21 — production
    #     fits RAW frames; the dark level comes from ccd_characteristics)
    is_reference_mode: bool
    andor_camera_info: AndorCameraInfo
