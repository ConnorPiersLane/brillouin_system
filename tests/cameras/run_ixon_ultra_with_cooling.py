# run_ixon_ultra_with_cooling.py
from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
import numpy as np

def test_ixon_camera():
    print("=== Starting IxonUltra camera test (with cooling) ===")

    cam = IxonUltra(index=0, temperature=20.0, fan_mode="full")

    assert cam.is_opened(), "Camera failed to open."
    print("[OK] Camera is opened.")

    # --- Test exposure
    exposure_time = 0.1
    cam.set_exposure_time(exposure_time)
    actual_exp = cam.get_exposure_time()
    assert abs(actual_exp - exposure_time) < 0.01
    print(f"[OK] Exposure verified: {actual_exp}s")

    # --- Test EMCCD gain
    gain_value = 100
    cam.set_emccd_gain(gain_value)
    assert cam.get_emccd_gain() == gain_value
    print(f"[OK] Gain verified: {gain_value}")

    # --- Test ROI and binning
    cam.set_roi(100, 200, 50, 150)
    cam.set_binning(1, 1)
    assert cam.get_roi() == (100, 200, 50, 150)
    assert cam.get_binning() == (1, 1)
    print("[OK] ROI and binning verified.")

    # --- Frame shape check
    expected_shape = ((150 - 50), (200 - 100))
    shape = cam.get_frame_shape()
    assert shape == expected_shape
    print(f"[OK] Frame shape verified: {shape}")

    # --- Flip test
    cam.set_flip_image_horizontally(True)
    assert cam.get_flip_image_horizontally() is True
    print("[OK] Horizontal flip flag verified.")

    # --- VSS index test
    cam.set_vss_index(2)
    assert cam.get_vss_index() == 2
    print("[OK] VSS index verified.")

    # --- Preamp mode test
    cam.set_pre_amp_mode(5)
    assert cam.get_pre_amp_mode() == 5
    print("[OK] Preamp mode index verified.")

    # --- Snap image test
    frame = cam.snap()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == shape
    assert frame.dtype in [np.uint16, np.uint32, np.float64]
    print("[OK] Snap image test passed.")

    # --- Temp check
    temp = cam.cam.get_temperature()
    print(f"[OK] Temperature check: {temp:.2f} °C")

    cam.close()
    assert not cam.is_opened()
    print("[OK] Camera closed cleanly.")

    print("=== ✅ All IxonUltra tests (with cooling) passed ===")

if __name__ == "__main__":
    test_ixon_camera()
