import cv2
import glob
import os
from pathlib import Path
from detect_pupil import detect_pupil
from eye_tracking.pupil_detection_helpers import mask_white_outside_radius


def test_pupil_detector(image_dir="./", pattern="pair_*.png", threshold_value=60):
    paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if not paths:
        print(f"No test images found in {image_dir} matching {pattern}")
        return

    output_dir = Path(image_dir) / "pupil_detections"
    output_dir.mkdir(exist_ok=True)

    for path in paths:
        print(f"\nProcessing: {path}")
        image = cv2.imread(path)
        if image is None:
            print(f"⚠️ Could not read {path}")
            continue
        # image = mask_white_outside_radius(image)

        result, binary = detect_pupil(image, threshold_value=threshold_value, debug=False)

        cv2.imshow("Binary Mask", binary)

        if result is None:
            print("❌ No pupil detected.")
            key = cv2.waitKey(0)
            if key == 27:  # ESC to exit
                break
            continue

        cx, cy, MA, ma, angle = result
        print(f"✅ Pupil found at ({cx:.1f}, {cy:.1f}), angle={angle:.1f}°, axes=({MA:.1f}, {ma:.1f})")

        out = image.copy()
        cv2.ellipse(out, ((cx, cy), (MA, ma), angle), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        cv2.imshow("Detection", out)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # You can tune threshold_value here
    test_pupil_detector(image_dir="right", pattern="pair_*.png", threshold_value=30)
