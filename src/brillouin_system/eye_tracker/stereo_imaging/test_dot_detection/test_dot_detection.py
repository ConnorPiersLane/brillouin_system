import cv2

from brillouin_system.eye_tracker.stereo_imaging.detect_dot_centroid import detect_dot_with_blob

# Load the test image
img_path = "pair_0001_left.png"
gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Run the centroid detection
centroid = detect_dot_with_blob(gray)

# Print and visualize result
if centroid is not None:
    cx, cy = centroid
    print(f"Detected centroid at: ({cx:.2f}, {cy:.2f})")

    # Draw the detected point
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    cv2.imshow("Detected Dot", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No dot detected.")
