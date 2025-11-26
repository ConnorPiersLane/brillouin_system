import cv2
import numpy as np
from matplotlib import pyplot as plt

from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import eye_tracker_config
from brillouin_system.eye_tracker.eye_tracker_helpers import draw_ellipse_rgb, gray_to_rgb
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter import EllipseFitter
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter_helpers import (
    find_pupil_ellipse_with_flooding, PupilImgType, extract_roi
)


def floodfill_and_show(path: str, threshold: int = 10):
    """
    Run pupil detection using the ellipse-fitting pipeline and visualize each processing step.

    Steps:
        1. Load an image in grayscale mode.
        2. Detect the pupil using `find_pupil_ellipse_with_flooding()` for each stage to visualize:
           ORIGINAL, BINARY, FLOODFILLED, CONTOUR.
        3. Display the four images side by side.
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

    # 2) Run detector once to get the ellipse (use fastest stage to avoid extra copies)

    img, x_clip, y_clip = extract_roi(img=img,
                        roi_center_xy=(500,500),
                        roi_width_height=(800,500))

    result_for_ellipse = find_pupil_ellipse_with_flooding(
        img, threshold=threshold, frame_to_be_returned=PupilImgType.FLOODFILLED
    )

    ellipse_params = None
    if result_for_ellipse.ellipse is not None:
        e = result_for_ellipse.ellipse
        ellipse_params = ((e.cx, e.cy), (e.major, e.minor), e.angle_deg)

    # 3) Collect each visualization stage
    stages = [
        (PupilImgType.ORIGINAL,   "Original gray"),
        (PupilImgType.BINARY,     "Binary (INV)"),
        (PupilImgType.FLOODFILLED,"Flood-filled"),
        (PupilImgType.CONTOUR,    "Largest component"),
    ]

    images = []
    for stype, _title in stages:
        r = find_pupil_ellipse_with_flooding(
            img, threshold=threshold, frame_to_be_returned=stype
        )
        images.append((r.pupil_img, _title))

    # 4) Build final visualization with ellipse drawn on ORIGINAL (for clarity)
    # We already have ORIGINAL from images[0]
    original_vis = images[0][0]
    # Ensure 3-channel for drawing helper
    vis_rgb = draw_ellipse_rgb(gray_to_rgb(original_vis), ellipse=result_for_ellipse.ellipse)

    # 5) Plot
    plt.figure(figsize=(20, 5))
    for i, (im, title) in enumerate(images, start=1):
        plt.subplot(1, 5, i)
        plt.title(title)
        plt.imshow(im, cmap='gray')
        plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.title("Fitted ellipse (green)")
    # draw_ellipse_rgb returns BGR; convert to RGB for matplotlib
    plt.imshow(cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    if ellipse_params is None:
        print("Warning: Could not fit ellipse (no contour or too few points).")

    return ellipse_params


if __name__ == "__main__":
    params = floodfill_and_show("left/pair_0000_left.png", threshold=8)
    print("Ellipse:", params)
