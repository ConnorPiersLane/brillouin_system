from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.background_image import ImageStatistics
from brillouin_system.utils.calibration import CalibrationResults
from brillouin_system.my_dataclasses.camera_settings import CameraSettings
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.my_dataclasses.zaber_position import ZaberPosition

@dataclass
class MeasurementPoint:
    is_reference_mode: bool
    do_live_fitting: bool
    frame: np.ndarray
    bg_frame: ImageStatistics | None
    fitting_results: FittedSpectrum
    zaber_position: ZaberPosition | None
    camera_settings: CameraSettings
    mako_image: np.ndarray | None

@dataclass
class MeasurementSeries:
    measurements: list[MeasurementPoint]
    calibration: CalibrationResults