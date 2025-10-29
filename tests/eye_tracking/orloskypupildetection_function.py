import cv2
import numpy as np

def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))

def get_darkest_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = (0, 0)

    for y in range(20, gray.shape[0] - 20, 10):
        for x in range(20, gray.shape[1] - 20, 10):
            patch = gray[y:y + 20, x:x + 20]
            patch_sum = np.sum(patch)
            if patch_sum < min_sum:
                min_sum = patch_sum
                darkest_point = (x + 10, y + 10)
    return darkest_point

def apply_binary_threshold(image, darkest_val, added_threshold):
    _, thresh = cv2.threshold(image, darkest_val + added_threshold, 255, cv2.THRESH_BINARY_INV)
    return thresh

def mask_outside_square(image, center, size):
    mask = np.zeros_like(image)
    x, y = center
    half = size // 2
    mask[max(0, y - half):y + half, max(0, x - half):x + half] = 255
    return cv2.bitwise_and(image, mask)

def filter_contours(contours, area_thresh=1000, ratio_thresh=3, image_shape=None, center_tolerance=0.30):
    """
    Filters contours based on area, aspect ratio, and proximity to image center.

    Args:
        contours: list of contours.
        area_thresh: minimum area threshold.
        ratio_thresh: maximum aspect ratio.
        image_shape: (height, width) of the image.
        center_tolerance: fraction of image size within which the ellipse must lie.

    Returns:
        best contour or None
    """
    best = None
    max_area = 0

    if image_shape:
        h, w = image_shape
        cx, cy = w // 2, h // 2
        tol_x, tol_y = int(w * center_tolerance), int(h * center_tolerance)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_thresh and len(cnt) >= 5:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            ratio = max(w_box / h_box, h_box / w_box)
            if ratio < ratio_thresh:
                # Check that the center of the ellipse is near the image center
                ellipse = cv2.fitEllipse(cnt)
                ex, ey = ellipse[0]
                if not image_shape or (abs(ex - cx) < tol_x and abs(ey - cy) < tol_y):
                    if area > max_area:
                        max_area = area
                        best = cnt
    return best

def detect_pupil(image_path, debug=False, display=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")

    image = crop_to_aspect_ratio(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    darkest_point = get_darkest_area(image)
    darkest_val = gray[darkest_point[1], darkest_point[0]]

    ellipses = []
    for t in [5, 15, 25]:
        thresh = apply_binary_threshold(gray, darkest_val, t)
        thresh = mask_outside_square(thresh, darkest_point, 250)
        dilated = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = filter_contours(contours, image_shape=gray.shape)
        if best_cnt is not None and len(best_cnt) >= 5:
            ellipse = cv2.fitEllipse(best_cnt)
            ellipses.append((ellipse, cv2.contourArea(best_cnt)))

    if not ellipses:
        return None

    best_ellipse = max(ellipses, key=lambda x: x[1])[0]

    if display:
        output = image.copy()
        cv2.ellipse(output, best_ellipse, (0, 255, 0), 2)
        cv2.imshow("Pupil Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if debug:
        print(f"Detected pupil ellipse: {best_ellipse}")

    return best_ellipse
