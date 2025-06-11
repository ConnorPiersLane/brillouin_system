from dataclasses import dataclass

import numpy as np



class ImageStatistics:
    def __init__(self, images: list[np.ndarray] | np.ndarray):
        self.mean_image: np.ndarray = np.mean(images, axis=0)
        self.std_image: np.ndarray = np.std(images, axis=0)
        self.n_images: list[np.ndarray] | np.ndarray = images

@dataclass
class BackGroundImage:
    dark_image: ImageStatistics | None
    bg_image: ImageStatistics | None