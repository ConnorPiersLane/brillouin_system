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

        print("\n=== Gain Range ===")
        print("Gain range:", cam.min_max_gain())

        print("\n=== Exposure Range ===")
        print("Exposure range (µs):", cam.min_max_exposure_time())

        print("\n=== Gamma Range ===")
        cam.set_pixel_format('Mono16')
        print("Gamma range:", cam.min_max_gamma())
        cam.set_pixel_format('Mono8')
        print("Gamma range:", cam.min_max_gamma())


        print("\n=== ROI Native ===")
        sensor_w, sensor_h = cam.get_sensor_size()
        cam.set_roi_native(0, 0, sensor_w // 2, sensor_h // 2)
        print("Set half ROI:", cam.get_roi_native())

        print("\n=== Gain ===")
        gain_value = cam.get_gain()
        cam.set_gain(gain_value + 1.0 if gain_value + 1.0 < 10 else gain_value)
        print("New gain:", cam.get_gain())

        print("\n=== Gain ===")
        gain_value = cam.get_gain()
        cam.set_gain(0)
        print("New gain:", cam.get_gain())

        print("\n=== Exposure ===")
        exposure_value = cam.get_exposure_time()
        cam.set_exposure_time(exposure_value + 5000 if exposure_value + 5000 < 1000000 else exposure_value)
        print("New exposure time:", cam.get_exposure_time())

        print("\n=== Gamma ===")
        try:
            cam.set_gamma(3.0)
            print("Gamma:", cam.get_gamma())
        except Exception as e:
            print("Gamma not supported:", e)
        print("\n=== Gamma Range ===")
        try:
            print("Gamma range:", cam.min_max_gamma())
        except Exception as e:
            print("Gamma range not available:", e)

        print("\n=== Acquire Image ===")
        # Reset acquisition mode and trigger to default (SingleFrame, trigger off)
        cam.start_single_frame_mode()
        img = cam.acquire_image()
        print("Image shape:", img.shape)
        cv2.imshow("Single Frame", cv2.convertScaleAbs(img, alpha=255.0 / 65535.0))
        cv2.waitKey(500)
        cv2.destroyAllWindows()

        print("\n=== Software Triggered Snap Stream ===")
        cam.start_software_stream()
        for i in range(3):
            img = cam.software_snap_while_stream()
            print(f"Frame {i + 1} shape:", img.shape)
            cv2.imshow("Stream Frame", cv2.convertScaleAbs(img, alpha=255.0 / 65535.0))
            cv2.waitKey(100)
        cam.end_software_stream()
        cv2.destroyAllWindows()

        print("\n=== Pixel Format Change Test ===")
        available_formats = cam.get_available_pixel_formats()
        print("Available formats:", available_formats)

        for fmt in available_formats:
            try:
                print(f"Setting pixel format to: {fmt}")
                cam.set_pixel_format(fmt)
                current = cam.get_pixel_format()
                if current == fmt:
                    print(f"✔️  Confirmed pixel format: {current}")
                else:
                    print(f"⚠️  Mismatch! Expected '{fmt}', got '{current}'")
            except Exception as e:
                print(f"❌ Failed to set format '{fmt}': {e}")

    finally:
        print("\n=== Shutting down ===")
        cam.shutdown()





if __name__ == "__main__":
    test_all()
