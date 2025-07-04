from PyQt5.QtGui import QImage, QPixmap
import numpy as np

def numpy_array_to_pixmap(array: np.ndarray) -> QPixmap:
    if array.ndim != 2:
        raise ValueError("Expected 2D grayscale image array")

    # Clean NaNs/Infs
    array = np.nan_to_num(array)

    # Normalize to 0–255
    array_min = array.min()
    array_max = array.max()
    ptp = array_max - array_min
    if ptp == 0:
        ptp = 1e-6  # avoid division by zero

    norm = ((array - array_min) / ptp * 255).astype(np.uint8)

    h, w = norm.shape
    qimg = QImage(norm.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)
