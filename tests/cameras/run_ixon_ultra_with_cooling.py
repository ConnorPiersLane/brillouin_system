from brillouinDAQ.devices.cameras.for_brillouin_signal.ixonUltra import IxonUltra


import numpy as np

# Be aware, the camera will cool down and heat up again. Do not repeat this test if it is not necessary.
# For simple tests, run the test without cooling
def test_ixon_camera():
    print("=== Starting IxonUltra camera test ===")

    # Create camera instance
    cam = IxonUltra(index=0, temperature=-10.0, fan_mode="full")

    # ---- Check camera is opened
    assert cam.is_opened(), "Camera failed to open."
    print("[OK] Camera is opened.")

    # ---- Set and verify exposure
    exposure_time = 0.1
    cam.set_exposure_time(exposure_time)
    actual_exp = cam.cam.get_exposure()
    assert abs(actual_exp - exposure_time) < 0.01, f"Exposure mismatch: set {exposure_time}, got {actual_exp}"
    print(f"[OK] Exposure set and verified: {actual_exp} s")

    # ---- Set and verify gain
    gain_value = 100
    cam.set_gain(gain_value)
    actual_gain, _ = cam.cam.get_EMCCD_gain()
    assert actual_gain == gain_value, f"Gain mismatch: set {gain_value}, got {actual_gain}"
    print(f"[OK] Gain set and verified: {actual_gain}")

    # ---- Set and verify ROI and binning
    roi_x_start, roi_x_end = 100, 200
    roi_y_start, roi_y_end = 50, 150
    hbin, vbin = 1, 1
    cam.set_roi(roi_x_start, roi_x_end, roi_y_start, roi_y_end)
    cam.set_binning(hbin, vbin)
    roi_params = cam.cam.get_image_mode_parameters()
    expected_roi = (roi_x_start, roi_x_end, roi_y_start, roi_y_end, hbin, vbin)
    assert roi_params == expected_roi, f"ROI mismatch: set {expected_roi}, got {roi_params}"
    print(f"[OK] ROI and binning set and verified: {roi_params}")

    # ---- Get and verify frame shape
    expected_width = (roi_x_end - roi_x_start) // hbin
    expected_height = (roi_y_end - roi_y_start) // vbin
    shape = cam.get_frame_shape()
    assert shape == (expected_height, expected_width), f"Shape mismatch: expected {expected_height}x{expected_width}, got {shape}"
    print(f"[OK] Frame shape verified: {shape}")

    # ---- Snap and check output shape/type
    frame = cam.snap()
    assert isinstance(frame, np.ndarray), "Snap did not return a NumPy array"
    assert frame.shape == shape, f"Snap shape mismatch. Expected {shape}, got {frame.shape}"
    assert frame.dtype in [np.uint16, np.uint32, np.float64], f"Unexpected data type: {frame.dtype}"
    print("[OK] Snap successful and shape matches ROI.")

    # ---- Temperature sanity check
    temp = cam.cam.get_temperature()
    print(f"[OK] Camera temperature check: {temp:.2f} °C")

    # ---- Close and check
    cam.close()
    assert not cam.is_opened(), "Camera failed to close properly."
    print("[OK] Camera closed successfully.")

    print("=== ✅ All IxonUltra camera tests passed ===")

if __name__ == "__main__":
    test_ixon_camera()
