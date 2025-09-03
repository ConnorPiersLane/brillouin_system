import numpy as np

from brillouin_system.my_dataclasses.background_image import ImageStatistics


def subtract_background(frame: np.ndarray, bg_frame: ImageStatistics) -> np.ndarray:
    if bg_frame is None:
        print("[subtract_background] no background frame available")
        return frame

    result = frame - bg_frame.median_image
    result = np.clip(result, 0, None)
    return result

def subtract_darknoise(frame: np.ndarray, darknoise_frame: ImageStatistics) -> np.ndarray:
    if darknoise_frame is None:
        print("[subtract_darknoise] no darknoise frame available")
        return frame

    result = frame - darknoise_frame.mean_image
    result = np.clip(result, 0, None)
    return result