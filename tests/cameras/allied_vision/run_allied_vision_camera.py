import time

from brillouin_system.devices.cameras.allied.allied_vision_camera import AlliedVisionCamera


# Replace with actual import path
def frame_handler(frame):
    try:
        image = frame.as_numpy_ndarray()
        print(f"Frame received | Shape: {image.shape}")
    except Exception as e:
        print(f"Frame handler error: {e}")

def main():
    cam = AlliedVisionCamera()
    print(dir(cam))

    # ---- Test Gain ----
    expected_gain = 10
    cam.set_gain(expected_gain)
    actual_gain = cam.get_gain()
    print(f"[TEST] Set gain: {expected_gain} dB | Read gain: {actual_gain} dB")
    if abs(expected_gain - actual_gain) > 0.1:
        print("❌ [ERROR] Gain mismatch!")

    # ---- Test Exposure ----
    expected_exposure = 20000
    cam.set_exposure(expected_exposure)
    actual_exposure = cam.get_exposure()
    print(f"[TEST] Set exposure: {expected_exposure} µs | Read exposure: {actual_exposure} µs")
    if abs(expected_exposure - actual_exposure) > 10:
        print("❌ [ERROR] Exposure mismatch!")


    # ---- Test Auto Exposure ----
    print("[TEST] Setting Auto Exposure to 'Continuous'...")
    cam.set_auto_exposure("Continuous")
    auto_exposure_mode = cam.get_auto_exposure()
    print(f"[TEST] Auto Exposure Mode: {auto_exposure_mode}")
    if auto_exposure_mode != "Continuous":
        print(f"❌ [ERROR] Auto Exposure not set correctly! Expected 'Continuous', got '{auto_exposure_mode}'")
    else:
        print("✅ [TEST] Auto Exposure successfully set to 'Continuous'")

    print("[TEST] Disabling Auto Exposure...")
    cam.set_auto_exposure("Off")
    auto_exposure_mode = cam.get_auto_exposure()
    print(f"[TEST] Auto Exposure Mode after disable: {auto_exposure_mode}")
    if auto_exposure_mode != "Off":
        print(f"❌ [ERROR] Failed to disable Auto Exposure! Expected 'Off', got '{auto_exposure_mode}'")
    else:
        print("✅ [TEST] Auto Exposure successfully disabled")

    # ---- Test ROI ----
    roi_settings = {"OffsetX": 0, "OffsetY": 0, "Width": 800, "Height": 800}
    cam.set_roi(**roi_settings)
    actual_roi = cam.get_roi()
    print(f"[TEST] Set ROI: {roi_settings} | Read ROI: {actual_roi}")
    for key, expected in roi_settings.items():
        actual = actual_roi.get(key)
        if expected != actual:
            print(f"❌ [ERROR] ROI mismatch on {key}: expected {expected}, got {actual}")

    # ---- Max ROI Test ----
    max_roi = cam.get_max_roi()
    print(f"[TEST] Max ROI: {max_roi}")
    cam.set_roi(**max_roi)
    actual_roi = cam.get_roi()
    print(f"[TEST] ROI after setting max: {actual_roi}")


    # ---- Stream Test ----
    print("[TEST] Starting stream for 10 seconds...")
    cam.start_stream(frame_handler)
    time.sleep(1)

    # ---- Snap Test ----
    print("[TEST] Capturing a single frame (snap)...")
    img = cam.snap()
    print(f"[TEST] Snap image shape: {img}")

    # ---- Clean up ----
    cam.stop_stream()
    cam.close()

if __name__ == "__main__":
    main()
