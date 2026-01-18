import cv2
import numpy as np

def mask_white_outside_radius(image, radius=500, center=None):
    """
    Turn every pixel outside a given circle white.
    Works with grayscale or color images.

    Args:
        image (np.ndarray): Input grayscale or BGR image.
        radius (int): Radius of circle to keep.
        center (tuple or None): (x, y) center of circle. Defaults to image center.

    Returns:
        np.ndarray: Image with outside pixels turned white (255).
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    # 1️⃣ Make mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # 2️⃣ Fill everything outside with white
    if len(image.shape) == 2:
        # grayscale
        output = image
        output[mask == 0] = 255
    else:
        # color (BGR)
        output = image.copy()
        output[mask == 0] = (255, 255, 255)

    return output
