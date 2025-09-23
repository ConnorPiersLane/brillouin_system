from dataclasses import dataclass, field

import numpy as np

# Idea: [Dual Camera Proxy] -> RawFrames -> [Object Detection] -> DetectedObjects
# -> [PostProcessing] -> EyePositionResults -> [ImageRenderer] -> RenderedImages






@dataclass
class RawFrames:
    left_frame: np.ndarray
    right_frame: np.ndarray
    msg: dict


@dataclass
class DetectedObjects:
    left_pupil_center: np.ndarray = field(default=None)
    right_pupil_center: np.ndarray = field(default=None)
    left_glint_position: np.ndarray = field(default=None)
    right_glint_position: np.ndarray = field(default=None)

@dataclass
class EyePositionResults:
    pupil_position: np.ndarray = field(default=None)
    cornea_center: np.ndarray = field(default=None)

@dataclass
class EyeTrackerResults:
    left_frame: np.ndarray
    right_frame: np.ndarray
