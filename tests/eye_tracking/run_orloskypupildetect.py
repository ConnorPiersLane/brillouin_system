from pathlib import Path

from orloskypupildetection_function import detect_pupil  # adjust if your filename differs
import sys
import os

# === CONFIGURE HERE ===
IMAGE_FOLDER = "/Users/margaretferris/Desktop/dummyeye8/left"
base = Path(__file__).resolve().parent  # folder of this test script
IMAGE_FOLDER = base / "right"
img_left2 = base / "left" / "pair_0001_left.png"
img_right1 = base / "right" / "pair_0000_right.png"
img_right2 = base / "right" / "pair_0001_right.png"
DISPLAY = True
DEBUG = False

def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"[ERROR] Folder not found: {IMAGE_FOLDER}")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("[ERROR] No images found in folder.")
        return

    for image_file in sorted(image_files):
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        print(f"\n[INFO] Processing: {image_path}")
        ellipse = detect_pupil(image_path, debug=DEBUG, display=DISPLAY)
        if ellipse:
            center, axes, angle = ellipse
            print(f"[RESULT] Pupil center: {center}, Axes: {axes}, Angle: {angle}")
        else:
            print("[RESULT] No pupil detected.")

if __name__ == "__main__":
    main()
