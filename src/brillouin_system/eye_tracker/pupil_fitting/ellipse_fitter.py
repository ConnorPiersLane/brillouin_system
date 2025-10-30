

import numpy as np

from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter_helpers import PupilEllipse, find_pupil_ellipse_with_flooding


class EllipseFitter:
    """
    Holds left/right configs (from TOML via your config accessors) and provides:
      - find_pupil_left(image)
      - find_pupil_right(image)
    Both call a single internal algorithm _find_pupil(image, cfg).

    Configs are ALWAYS used (no ad-hoc overrides here). Call .refresh() to re-pull
    latest values if you edit the TOML or change them via your config GUI.
    """

    def __init__(self, binary_threshold_left: int = 20, binary_threshold_right: int = 20) -> None:

        self.binary_threshold_left = binary_threshold_left
        self.binary_threshold_right = binary_threshold_right


    def set_binary_thresholds(self, binary_threshold_left: int = 20, binary_threshold_right: int = 20) -> None:
        self.binary_threshold_left = binary_threshold_left
        self.binary_threshold_right = binary_threshold_right

    # ---- Public API ----
    def find_pupil_left(self, image: np.ndarray) -> PupilEllipse:
        return find_pupil_ellipse_with_flooding(img=image, threshold=self.binary_threshold_left)

    def find_pupil_right(self, image: np.ndarray) -> PupilEllipse:
        return find_pupil_ellipse_with_flooding(img=image, threshold=self.binary_threshold_right)

