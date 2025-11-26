

import numpy as np

from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter_helpers import PupilEllipse, \
    find_pupil_ellipse_with_flooding, PupilImgType, extract_roi

RETURN_FRAME_MAPPING = {
    "original": PupilImgType.ORIGINAL,
    "binary": PupilImgType.BINARY,
    "floodfilled": PupilImgType.FLOODFILLED,
    "contour": PupilImgType.CONTOUR,
}

class EllipseFitter:
    """
    Holds left/right configs (from TOML via your config accessors) and provides:
      - find_pupil_left(image)
      - find_pupil_right(image)
    Both call a single internal algorithm _find_pupil(image, cfg).

    Configs are ALWAYS used (no ad-hoc overrides here). Call .refresh() to re-pull
    latest values if you edit the TOML or change them via your config GUI.
    """

    def __init__(self, config: EyeTrackerConfig) -> None:

        self.config: EyeTrackerConfig = config

    def set_config(self, config: EyeTrackerConfig):
        self.config = config


    # ---- Public API ----


    def find_pupil_left(self, image: np.ndarray) -> PupilEllipse:

        if self.config.apply_roi:
            image = extract_roi(img=image,
                                roi_center_xy=self.config.roi_left_center_xy,
                                roi_width_height=self.config.roi_left_width_height)

        return find_pupil_ellipse_with_flooding(img=image,
                                                threshold=self.config.binary_threshold_left,
                                                frame_to_be_returned=RETURN_FRAME_MAPPING[self.config.frame_returned])

    def find_pupil_right(self, image: np.ndarray) -> PupilEllipse:
        if self.config.apply_roi:
            image = extract_roi(img=image,
                                roi_center_xy=self.config.roi_right_center_xy,
                                roi_width_height=self.config.roi_right_width_height)

        return find_pupil_ellipse_with_flooding(img=image,
                                                threshold=self.config.binary_threshold_right,
                                                frame_to_be_returned=RETURN_FRAME_MAPPING[self.config.frame_returned])

