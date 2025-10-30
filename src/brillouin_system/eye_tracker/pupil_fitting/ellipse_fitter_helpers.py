import cv2
import numpy as np
from dataclasses import dataclass
from typing import Literal
from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D


@dataclass
class PupilEllipse:
    """
    Container for pupil-detection results.

    Attributes:
        original_img: Original input image (grayscale, uint8).
        binary_img: Binary inverse threshold result (pupil dark -> white).
        floodfilled_img: Largest connected component mask (uint8, 0/255).
        ellipse: Fitted ellipse parameters or None if no valid fit. The ellipse fields are:
                 - cx, cy: Center coordinates (float)
                 - major, minor: Major/minor axis lengths (float)
                 - angle_deg: Rotation in degrees (float, OpenCV-style: angle of the major axis)
    """
    original_img: np.ndarray          # input grayscale
    binary_img: np.ndarray            # thresholded binary
    floodfilled_img: np.ndarray       # largest component mask
    ellipse: Ellipse2D | None     # fitted ellipse params (or None)


def _ensure_u8(img: np.ndarray) -> np.ndarray:
    """Normalize any numeric single- or 3-channel image to uint8 (keeps shape)."""
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    m, M = np.min(img), np.max(img)
    if M > m:
        scaled = (img - m) * (255.0 / (M - m))
        return np.ascontiguousarray(scaled.astype(np.uint8))
    # constant image
    return np.zeros_like(img, dtype=np.uint8)


def find_pupil_ellipse_with_flooding(img: np.ndarray, threshold: int) -> PupilEllipse :
    """
    Fast pupil ellipse detection via binary inversion, flood fill, largest component, and ellipse fit.

    Pipeline:
        1) Binary inverse threshold (pupil dark -> white).
        2) Flood fill from all four corners to mark and remove background.
        3) Keep the largest connected component as the pupil candidate.
        4) Fit an ellipse to the candidate contour (if >= 5 points).

    This implementation is optimized for real-time usage: no blurs, morphology, or extra copies.

    Args:
        img: Input grayscale image (any numeric dtype). Will be coerced to uint8 if needed.
        threshold: Threshold value in [0..255] for cv2.THRESH_BINARY_INV.
        Choose level between dark pupil and surrounding white

    Returns:
        PupilEllipse:
            - original_img: uint8 grayscale input (post-normalization if needed)
            - binary_img: binary inverse threshold result
            - floodfilled_img: largest component mask (0/255)
            - color_img: BGR visualization with ellipse in green if found
            - ellipse: Ellipse2D or None

    Notes:
        - If no connected component is found (beyond background) or too few contour points (<5),
          ellipse will be None and color_img will be just a BGR version of the input.

    """
    img = _ensure_u8(img)

    # Step 1: Binary inverse threshold — pupil dark → white
    _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Flood fill from corners to remove background
    h, w = bw.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(bw, mask, (0, 0), 128)
    cv2.floodFill(bw, mask, (w - 1, 0), 128)
    cv2.floodFill(bw, mask, (0, h - 1), 128)
    cv2.floodFill(bw, mask, (w - 1, h - 1), 128)
    bw[bw == 128] = 0  # remove filled background

    # Step 3: Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels <= 1:
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return PupilEllipse(img, bw, bw, None)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    largest_mask = np.array(labels == largest_label, dtype=np.uint8) * 255

    # Step 4: Fit ellipse
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipse_obj = None

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (major, minor), angle_deg = ellipse
            ellipse_obj = Ellipse2D(cx, cy, major, minor, angle_deg)

    return PupilEllipse(img, bw, largest_mask, ellipse_obj)
