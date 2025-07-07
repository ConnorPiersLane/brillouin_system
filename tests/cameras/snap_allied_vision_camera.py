import cv2
import numpy as np
from brillouin_system.devices.cameras.allied.allied_vision_camera import AlliedVisionCamera


def main():
    print("[DEBUG] Initializing Allied Vision camera...")
    cam = AlliedVisionCamera()

    try:
        # Optional: set ROI or exposure
        cam.set_exposure(5000)  # µs
        cam.set_gain(10)

        print("[DEBUG] Capturing one image...")
        image = cam.snap()

        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        image = np.ascontiguousarray(image)

        # Normalize for display
        norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        print(f"[DEBUG] Image shape: {norm.shape}, dtype: {norm.dtype}")

        # Display image
        cv2.imshow("Allied Vision Snap", norm)
        print("[DEBUG] Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        print("[DEBUG] Closing camera...")
        cam.close()

if __name__ == "__main__":
    main()