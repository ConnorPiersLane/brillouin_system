
import cv2
import numpy as np


def make_blob_detector(
    min_area: float = 15.0,
    max_area: float = 5000.0,
    min_circularity: float = 0.6,
    dark: bool = True,
):
    """Blob detector configured for a small circular calibration dot."""
    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor = True
    p.blobColor = 0 if dark else 255
    p.filterByArea = True
    p.minArea = float(min_area)
    p.maxArea = float(max_area)
    p.filterByCircularity = True
    p.minCircularity = float(min_circularity)
    p.filterByConvexity = True
    p.minConvexity = 0.8
    p.filterByInertia = False
    return cv2.SimpleBlobDetector_create(p)


def _subpixel_centroid(gray: np.ndarray, u: float, v: float, win: int = 7,
                       dark: bool = True) -> tuple[float, float]:
    """Refine a dot center with an intensity-weighted centroid around (u, v)."""
    h, w = gray.shape[:2]
    x0, x1 = int(round(u)) - win, int(round(u)) + win + 1
    y0, y1 = int(round(v)) - win, int(round(v)) + win + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return float(u), float(v)
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    weights = (patch.max() - patch) if dark else (patch - patch.min())
    s = weights.sum()
    if s <= 0:
        return float(u), float(v)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float((weights * xs).sum() / s), float((weights * ys).sum() / s)


def detect_dot(
    gray: np.ndarray,
    detector=None,
    near: tuple[float, float] | None = None,
    subpixel: bool = True,
    dark: bool = True,
) -> tuple[float, float] | None:
    """
    Detect a single calibration dot.

    - `detector`: reuse a detector from make_blob_detector() (faster); default params otherwise.
    - `near`: if several blobs are found, prefer the one closest to this (u, v)
      (temporal consistency across frames); otherwise the smallest blob is taken.
    - `subpixel`: refine the winning blob with an intensity-weighted centroid.
    """
    if detector is None:
        detector = make_blob_detector(dark=dark)
    keypoints = detector.detect(gray)
    if not keypoints:
        return None
    if near is not None and len(keypoints) > 1:
        kp = min(keypoints, key=lambda k: (k.pt[0] - near[0]) ** 2 + (k.pt[1] - near[1]) ** 2)
    else:
        kp = min(keypoints, key=lambda k: k.size)
    u, v = float(kp.pt[0]), float(kp.pt[1])
    if subpixel:
        win = max(5, int(round(kp.size)))
        u, v = _subpixel_centroid(gray, u, v, win=win, dark=dark)
    return u, v


def detect_dot_with_blob(gray):
    """Backward-compatible wrapper (old callers)."""
    return detect_dot(gray)


def detect_dot_with_blob_dummy(gray):
    h, w = gray.shape[:2]
    return float(h / 2), float(w / 2)
