import cv2
import matplotlib.pyplot as plt
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter import EllipseFitter, draw_ellipse

# --- Define your test images ---
# Adjust paths as needed
from pathlib import Path

base = Path(__file__).resolve().parent  # folder of this test script
img_left1 = base / "left" / "pair_0000_left.png"
img_left2 = base / "left" / "pair_0001_left.png"
img_right1 = base / "right" / "pair_0000_right.png"
img_right2 = base / "right" / "pair_0001_right.png"


# --- Initialize fitter ---
ellipse_fitter = EllipseFitter()

# --- Load and process each frame ---
def process_and_show(img_path, side="left"):
    print(f"Processing {side} eye: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not load {img_path}")
        return

    if side == "left":
        e = ellipse_fitter.find_pupil_left(img)
    else:
        e = ellipse_fitter.find_pupil_right(img)

    # Draw ellipse overlay
    overlay = draw_ellipse(img, e)

    # Display results
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"{side} eye\n{e}")
    plt.axis("off")
    plt.show()

    print(f"→ Detected ellipse: {e}\n")

# --- Run tests ---
process_and_show(img_left1, "left")
process_and_show(img_left2, "left")
process_and_show(img_right1, "right")
process_and_show(img_right2, "right")
