import cv2
import numpy as np
from dataclasses import dataclass
from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D
from enum import Enum, auto


class PupilImgType(Enum):
    """Stages of the pupil detection pipeline."""
    ORIGINAL = auto()       # raw input or normalized grayscale
    BINARY = auto()         # after thresholding (pupil dark → white)
    FLOODFILLED = auto()    # after background flood-fill removal
    CONTOUR = auto()        # largest connected component / final contour mask

    def __str__(self):
        return self.name.lower()


@dataclass
class PupilEllipse:
    """
    Container for pupil-detection results.

    Attributes:
        pupil_img: Returned image at the requested stage.
        pupil_img_type: Which stage this image represents.
        ellipse: Fitted ellipse parameters or None if no valid fit.
                 Fields: (cx, cy, major, minor, angle_deg)
    """
    pupil_img: np.ndarray
    pupil_img_type: PupilImgType
    ellipse: Ellipse2D | None


def img_to_be_returned(
        pupil_img_type: PupilImgType,
        original_frame: np.ndarray | None,
        binary_img: np.ndarray | None,
        floodfilled_img: np.ndarray | None,
        contour_img: np.ndarray | None,
) -> tuple[np.ndarray, PupilImgType]:
    """
    Returns the requested image type from the pupil detection pipeline,
    with graceful fallback to the most processed available image.

    Returns:
        (image, actual_type): tuple containing the selected image and
        the PupilImgType that corresponds to it.
    """
    img_map = {
        PupilImgType.ORIGINAL: original_frame,
        PupilImgType.BINARY: binary_img,
        PupilImgType.FLOODFILLED: floodfilled_img,
        PupilImgType.CONTOUR: contour_img,
    }

    # Try the explicitly requested image first
    img = img_map.get(pupil_img_type)
    if img is not None:
        return img, pupil_img_type

    # Fallback order: most processed → least
    fallback_order = [
        (contour_img, PupilImgType.CONTOUR),
        (floodfilled_img, PupilImgType.FLOODFILLED),
        (binary_img, PupilImgType.BINARY),
        (original_frame, PupilImgType.ORIGINAL),
    ]

    for candidate, candidate_type in fallback_order:
        if candidate is not None:
            return candidate, candidate_type

    raise ValueError("No valid image available to return (all inputs were None).")

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


def find_pupil_ellipse_with_flooding(
    img: np.ndarray,
    threshold: int = 20,
    frame_to_be_returned: PupilImgType = PupilImgType.ORIGINAL
) -> PupilEllipse:
    """
    Fast pupil ellipse detection via binary inversion, flood fill,
    largest component, and ellipse fit.

    Returns a PupilEllipse whose `pupil_img` is whichever stage you request.
    """
    img = _ensure_u8(img)

    # Step 1: Binary inverse threshold — pupil dark → white
    _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

    # Only keep a copy of the binary if it's the requested output
    binary = bw.copy() if frame_to_be_returned == PupilImgType.BINARY else None

    # Step 2: Flood fill from corners to remove background (in-place on bw)
    h, w = bw.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(bw, mask, (0, 0), 128)
    cv2.floodFill(bw, mask, (w - 1, 0), 128)
    cv2.floodFill(bw, mask, (0, h - 1), 128)
    cv2.floodFill(bw, mask, (w - 1, h - 1), 128)
    bw[bw == 128] = 0  # remove background marked by 128

    # Step 3: Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8, ltype=cv2.CV_32S)
    ellipse_obj = None
    contour_img = None

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        # Build a mask for the largest component (0/255)
        contour_img = np.zeros_like(bw, dtype=np.uint8)
        contour_img[labels == largest_label] = 255

        # Step 4: Fit ellipse from the largest component
        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5:
                (cx, cy), (major, minor), angle_deg = cv2.fitEllipse(cnt)
                ellipse_obj = Ellipse2D(cx, cy, major, minor, angle_deg)

    # Decide which image to return
    out_img, out_type = img_to_be_returned(
        pupil_img_type=frame_to_be_returned,
        original_frame=img,
        binary_img=binary,
        floodfilled_img=bw,
        contour_img=contour_img,
    )

    return PupilEllipse(
        pupil_img=out_img,
        pupil_img_type=out_type,
        ellipse=ellipse_obj,
    )


