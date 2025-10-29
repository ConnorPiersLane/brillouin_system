import cv2
import numpy as np

def detect_pupil(image, threshold_value=60, min_radius_px=5, max_radius_px=None, debug=False):
    """
    Manual-threshold pupil detector. Returns (ellipse, binary_mask).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Apply Gaussian blur to smooth noise ---
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Manual threshold (invert so pupil = white in binary) ---
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # --- Morphological cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Find contours ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, binary

    img_area = gray.shape[0] * gray.shape[1]
    # filter by area
    cands = [c for c in contours if len(c) >= 5 and
             5e-4 * img_area < cv2.contourArea(c) < 0.5 * img_area]
    if not cands:
        return None, binary

    best = max(cands, key=cv2.contourArea)
    (cx, cy), (MA, ma), angle = cv2.fitEllipse(best)

    if ma > MA:
        MA, ma = ma, MA
        angle = (angle + 90.0) % 180.0

    # radius filtering
    rM, rm = MA / 2, ma / 2
    if max_radius_px is None:
        max_radius_px = 0.22 * min(gray.shape[:2])
    if rM > max_radius_px or rm > max_radius_px or rM < min_radius_px or rm < min_radius_px:
        return None, binary

    if debug:
        out = image.copy()
        cv2.ellipse(out, ((cx, cy), (MA, ma), angle), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.imshow("Binary Mask", binary)
        cv2.imshow("Detection", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (cx, cy, MA, ma, angle), binary
