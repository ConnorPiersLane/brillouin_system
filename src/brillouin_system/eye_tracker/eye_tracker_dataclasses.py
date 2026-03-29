import threading
from dataclasses import dataclass

import numpy as np

from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
from brillouin_system.eye_tracker.pupil_fitting.pupil3D import Pupil3D
from brillouin_system.eye_tracker.stereo_imaging.calibration_dataclasses import StereoCalibration


@dataclass
class EyeTrackerSettings:
    config: EyeTrackerConfig
    stereo_calibration: StereoCalibration



@dataclass
class EyeTrackerResultsForGui:
    pupil3D: Pupil3D | None
    cam_left_img: np.ndarray
    cam_right_img: np.ndarray


