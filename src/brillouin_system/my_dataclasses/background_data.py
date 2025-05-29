import numpy as np

from brillouin_system.my_dataclasses.camera_settings import CameraSettings



class ImageStatistics:
    def __init__(self, images: list[np.ndarray], camera_settings: CameraSettings):
        if not images:
            raise ValueError("Image list is empty.")

        self.mean_image = np.mean(images, axis=0)
        self.std_image = np.std(images, axis=0)
        self.n = len(images)
        self.camera_settings = camera_settings