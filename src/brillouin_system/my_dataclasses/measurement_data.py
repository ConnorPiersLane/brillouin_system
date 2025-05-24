from dataclasses import dataclass

import numpy as np

from brillouin_system.utils.calibration import CalibrationResults
from brillouin_system.my_dataclasses.camera_settings import CameraSettings
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.my_dataclasses.zaber_position import ZaberPosition

@dataclass
class MeasurementData:
    is_reference_mode: bool
    fitting_results: FittedSpectrum
    calibration: CalibrationResults
    zaber_position: ZaberPosition | None
    camera_settings: CameraSettings
    mako_image: np.ndarray | None
