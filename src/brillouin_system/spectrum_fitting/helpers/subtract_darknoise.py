import numpy as np

from brillouin_system.my_dataclasses.background_image import ImageStatistics


def subtract_darknoise(frame: np.ndarray, darknoise_frame: ImageStatistics) -> np.ndarray:
    if darknoise_frame is None:
        return frame

    result = frame - darknoise_frame.mean_image
    result = np.clip(result, 0, None)
    return result
