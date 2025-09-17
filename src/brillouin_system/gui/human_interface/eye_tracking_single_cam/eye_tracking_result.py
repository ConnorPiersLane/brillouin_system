from dataclasses import dataclass, field

import numpy as np


@dataclass
class EyeTrackingResult:
    img: np.ndarray
    rendered_eye: np.ndarray = field(default=None)