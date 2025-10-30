import cv2
import numpy as np
from matplotlib import pyplot as plt

from brillouin_system.eye_tracker.eye_tracker_helpers import draw_ellipse_rgb, gray_to_rgb
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter_helpers import (
    find_pupil_ellipse_with_flooding,
)



def floodfill_and_show(path: str, threshold: int = 20):
    """
    Run pupil detection using the ellipse-fitting pipeline and visualize each processing step.

    Steps:
        1. Load an image in grayscale mode.
        2. Detect the pupil using `find_pupil_ellipse_with_flooding()`.
        3. Display the intermediate binary, largest-component, and color results side by side.
        4. Return the fitted ellipse parameters (if found).

    Args:
        path (str): Path to the input image file (must be readable by OpenCV).
        threshold (int, optional): Binary inverse threshold value used by the pupil detector. Default = 20.

    Returns:
        tuple | None:
            ((cx, cy), (major_axis_length, minor_axis_length), angle_degrees),
            or None if no valid ellipse could be fitted.

    Raises:
        FileNotFoundError: If the image cannot be read from the provided path.
    """
    # 1) Read grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # 2) Run detector (returns PupilEllipse)
    pupil = find_pupil_ellipse_with_flooding(img, threshold=threshold)

    # 3) Build OpenCV-style ellipse tuple for compatibility with old code
    ellipse_params = None
    if pupil.ellipse is not None:
        e = pupil.ellipse
        ellipse_params = ((e.cx, e.cy), (e.major, e.minor), e.angle_deg)

    # 4) Visualize intermediate results
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title("Original gray")
    plt.imshow(pupil.original_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Binary (INV)")
    plt.imshow(pupil.binary_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Largest component")
    plt.imshow(pupil.floodfilled_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Fitted ellipse (green)")
    # Use the helper to draw on whichever image you want; here we use the original:

    vis = draw_ellipse_rgb(gray_to_rgb(pupil.floodfilled_img), ellipse=pupil.ellipse)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    if ellipse_params is None:
        print("Warning: Could not fit ellipse (no contour or too few points).")

    return ellipse_params


if __name__ == "__main__":
    params = floodfill_and_show("left/pair_0001_left.png", threshold=20)
    print("Ellipse:", params)
