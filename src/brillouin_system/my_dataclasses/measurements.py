from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.background_image import ImageStatistics
from brillouin_system.my_dataclasses.state_mode import StateMode
from brillouin_system.utils.calibration import CalibrationResults
from brillouin_system.my_dataclasses.camera_settings import CameraSettings
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.my_dataclasses.zaber_position import ZaberPosition

@dataclass
class MeasurementPoint:
    is_reference_mode: bool
    frame: np.ndarray  # Original frame, not subtracted
    state_mode: StateMode
    fitting_results: FittedSpectrum
    zaber_position: ZaberPosition | None
    mako_image: np.ndarray | None

@dataclass
class MeasurementSeries:
    measurements: list[MeasurementPoint]
    calibration: CalibrationResults