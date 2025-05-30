from dataclasses import dataclass

import numpy as np

from brillouin_system.my_dataclasses.camera_settings import CameraSettings


class ImageStatistics:
    def __init__(self, images: list[np.ndarray] | np.ndarray):
        self.mean_image: np.ndarray = np.mean(images, axis=0)
        self.std_image: np.ndarray = np.std(images, axis=0)
        self.n: int = len(images)

@dataclass
class BackGroundImage:
    dark_image: ImageStatistics
    bg_image: ImageStatistics
    camera_settings: CameraSettings