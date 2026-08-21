from dataclasses import dataclass

import numpy as np


@dataclass
class MeasurementPoint:
    frame_andor: np.ndarray  # Original frame, not subtracted
    lens_zaber_position: float
    time_stamp: float | None = None
