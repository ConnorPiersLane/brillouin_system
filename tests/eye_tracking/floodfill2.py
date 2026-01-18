import cv2
import numpy as np
from matplotlib import pyplot as plt

def floodfill_and_show(path):
    # Read grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Threshold (your current choice)
    _, bw = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)

    # Flood fill from a border pixel to remove white background
    h, w = bw.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    ff = bw.copy()
    cv2.floodFill(ff, mask, (0, 0), 128)  # mark background as 128
    ff[ff == 128] = 0                      # set background to 0 (black)

    # --- Keep only the largest connected component ---
    bin_ff = (ff > 0).astype(np.uint8) * 255
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_ff, connectivity=8)

    if numLabels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        largest_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        largest_mask = np.zeros_like(bin_ff)

    # --- Find external contour of the largest component ---
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Prepare color image for drawing
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    ellipse_drawn = False
    ellipse_params = None  # (center(x,y), (major, minor), angle_degrees)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        # cv2.fitEllipse requires at least 5 points
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)  # ((cx, cy), (major, minor), angle_deg)
            ellipse_params = ellipse
            # Draw ellipse in red
            cv2.ellipse(color_img, ellipse, (0, 0, 255), 2)
            ellipse_drawn = True

    # Show results
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title("Original binary")
    plt.imshow(bw, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Background removed")
    plt.imshow(ff, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Largest component")
    plt.imshow(largest_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Fitted ellipse (red)")
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Optionally return ellipse parameters for downstream use
    # ellipse_params format: ((cx, cy), (major_axis_length, minor_axis_length), angle_degrees)
    if not ellipse_drawn:
        print("Warning: Could not fit ellipse (no contour or too few points).")
    return ellipse_params

# Example:
params = floodfill_and_show("left/pair_0001_left.png")
print("Ellipse:", params)
