import time
import cv2

from brillouin_system.devices.cameras.flir.flir_cam import FLIRCamera


def test_all():
    cam = FLIRCamera()

    try:
        print("\n=== Camera Info ===")
        info = cam.get_camera_info()
        for k, v in info.items():
            print(f"{k}: {v}")

        print("\n=== Sensor Size ===")
        print("Sensor size:", cam.get_sensor_size())

        print("\n=== Resolution ===")
        print("Current resolution:", cam.get_resolution())

        print("\n=== ROI ===")
        cam.set_max_roi()
        print("Set max ROI:", cam.get_roi_native())

        print("\n=== ROI Native ===")
        sensor_w, sensor_h = cam.get_sensor_size()
        cam.set_roi_native(0, 0, sensor_w // 2, sensor_h // 2)
        print("Set half ROI:", cam.get_roi_native())

        print("\n=== Gain ===")
        gain_value = cam.get_gain()
        cam.set_gain(gain_value + 1.0 if gain_value + 1.0 < 10 else gain_value)
        print("New gain:", cam.get_gain())

        print("\n=== Exposure ===")
        exposure_value = cam.get_exposure_time()
        cam.set_exposure_time(exposure_value + 5000 if exposure_value + 5000 < 1000000 else exposure_value)
        print("New exposure time:", cam.get_exposure_time())

        print("\n=== Gamma ===")
        try:
            cam.set_gamma(1.0)
            print("Gamma:", cam.get_gamma())
        except Exception as e:
            print("Gamma not supported:", e)

        print("\n=== Acquire Image ===")
        img = cam.acquire_image()
        print("Image shape:", img.shape)
        cv2.imshow("Single Frame", cv2.convertScaleAbs(img, alpha=255.0/65535.0))
        cv2.waitKey(500)
        cv2.destroyAllWindows()

        print("\n=== Software Triggered Snap Stream ===")
        cam.start_software_stream()
        for i in range(3):
            img = cam.software_snap_while_stream()
            print(f"Frame {i+1} shape:", img.shape)
            cv2.imshow("Stream Frame", cv2.convertScaleAbs(img, alpha=255.0/65535.0))
            cv2.waitKey(100)
        cam.end_software_stream()
        cv2.destroyAllWindows()

    finally:
        print("\n=== Shutting down ===")
        cam.shutdown()

if __name__ == "__main__":
    test_all()
