from dataclasses import dataclass

import numpy as np


def generate_image_statistics_dataclass(images: list[np.ndarray] | np.ndarray):
    mean_image: np.ndarray = np.mean(images, axis=0)
    std_image: np.ndarray = np.std(images, axis=0)
    n_images: list[np.ndarray] | np.ndarray = images
    return ImageStatistics(
        mean_image=mean_image,
        std_image=std_image,
        n=len(n_images),
    )

@dataclass
class ImageStatistics:
    mean_image: np.ndarray
    std_image: np.ndarray
    n: int

@dataclass
class BackgroundImage:
    dark_image: ImageStatistics | None = None
    bg_image: ImageStatistics | None = None