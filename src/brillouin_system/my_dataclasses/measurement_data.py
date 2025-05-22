from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.camera_settings import CameraSettings
from brillouin_system.my_dataclasses.fitting_results import FittingResults
from brillouin_system.my_dataclasses.zaber_position import ZaberPosition

@dataclass
class MeasurementData:
    fitting_results: FittingResults
    zaber_position: ZaberPosition
    camera_settings: CameraSettings
    mako_image: np.ndarray
