import cv2
import numpy as np

def find_white_pupil_from_binary(binary_img,
                                 min_area_frac=5e-4,
                                 max_area_frac=0.03,
                                 min_radius_px=6,
                                 max_radius_px=None):
    """
    Detect a *white* pupil in a binary image (0/255).
    Assumes: outside lens is white (already masked), pupil is white, eyeball/body is black.

    Returns:
        (cx, cy, MA, ma, angle), debug_vis  OR  (None, debug_vis)
    """
    # Ensure single-channel 0/255
    if binary_img.ndim == 3:
        binary = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    else:
        binary = binary_img.copy()
    binary = (binary > 0).astype(np.uint8) * 255

    H, W = binary.shape
    img_area = H * W
    if max_radius_px is None:
        max_radius_px = int(0.22 * min(H, W))  # cap to avoid lens-like rings

    # Small cleanup (remove pepper noise in the white regions)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 1)

    # Invert -> eyeball becomes white; pupil becomes a *hole* inside it
    inv = 255 - binary

    # Find contours with hierarchy so we can access holes
    contours, hier = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hier is None:
        return None, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    hier = hier[0]

    best = None
    best_score = -1.0

    # Visual debug image
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        # Only consider child contours (holes inside a white region in 'inv')
        parent_idx = hier[i][3]
        if parent_idx == -1:
            continue  # not a hole → not the pupil

        if len(cnt) < 5:
            continue

        area = cv2.contourArea(cnt)
        if not (min_area_frac * img_area < area < max_area_frac * img_area):
            continue

        # Fit ellipse to candidate
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        rM, rm = MA * 0.5, ma * 0.5
        if rM < min_radius_px or rm < min_radius_px or rM > max_radius_px or rm > max_radius_px:
            continue

        # Circularity (1.0 = perfect circle). Higher is better.
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = (4.0 * np.pi * area) / (perim * perim)

        # Score: circularity * area favors a solid round blob
        score = circularity * area

        # Draw candidate (yellow)
        cv2.ellipse(vis, ((cx, cy), (MA, ma), angle), (0, 255, 255), 1)

        if score > best_score:
            best_score = score
            best = (cx, cy, MA, ma, angle)

    # Draw chosen (green) or return None
    if best is not None:
        cx, cy, MA, ma, angle = best
        cv2.ellipse(vis, ((cx, cy), (MA, ma), angle), (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255), -1)
        return best, vis

    return None, vis

if __name__ == "__main__":
    # Load your binary (0/255). If you only have grayscale, threshold first.
    img = cv2.imread("right/pair_0000_right.png", cv2.IMREAD_GRAYSCALE)

    # Example: simple manual threshold to create the binary (adjust as needed)
    _, binary = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)

    # Optional: mask lens rim before this (set outside pixels to white)

    result, vis = find_white_pupil_from_binary(binary)

    cv2.imshow("Binary", binary)
    cv2.imshow("Pupil candidates & best", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if result is None:
        print("No white pupil found.")
    else:
        cx, cy, MA, ma, angle = result
        print(f"Pupil: center=({cx:.1f},{cy:.1f}) axes=({MA:.1f},{ma:.1f}) angle={angle:.1f}")
