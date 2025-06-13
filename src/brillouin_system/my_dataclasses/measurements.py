from dataclasses import dataclass, field

import numpy as np


from brillouin_system.my_dataclasses.state_mode import StateMode
from brillouin_system.my_dataclasses.calibration import CalibrationData
from brillouin_system.my_dataclasses.zaber_position import ZaberPosition


@dataclass
class MeasurementSettings:
    n_measurements: int
    name: str = 'Unnamed'
    power_mW: float = 0.0
    move_axes: str = ''
    move_x_rel_um: float = 0.0
    move_y_rel_um: float = 0.0
    move_z_rel_um: float = 0.0

@dataclass
class MeasurementPoint:
    frame: np.ndarray  # Original frame, not subtracted
    zaber_position: ZaberPosition # tuple[float, float, float]
    mako_image: np.ndarray = field(default=None)

@dataclass
class MeasurementSeries:
    measurements: list[MeasurementPoint]
    state_mode: StateMode
    calibration_data: CalibrationData | None
    settings: MeasurementSettings