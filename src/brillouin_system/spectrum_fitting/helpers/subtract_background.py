import numpy as np


def subtract_background(frame: np.ndarray, bg_frame: np.ndarray) -> np.ndarray:
    result = frame - bg_frame
    result = np.clip(result, 0, None)
    return result