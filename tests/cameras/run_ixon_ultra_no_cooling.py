# run_ixon_ultra_no_cooling.py
from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
import numpy as np

def test_ixon_camera():
    print("=== Starting IxonUltra camera test (no cooling) ===")

    cam = IxonUltra(index=0, temperature="off", fan_mode="full")

    assert cam.is_opened(), "Camera failed to open."
    print("[OK] Camera is opened.")

    # --- Verbose mode and available info
    cam.list_amp_modes()
    print("[INFO] Current amp mode:", cam.get_amp_mode())

    print("[INFO] Available VSSpeeds:", cam.cam.get_all_vsspeeds())
    device_info = cam.cam.get_device_info()
    print("[INFO] Device Info:", device_info)

    # --- Gain test
    cam.set_fixed_pre_amp_mode(index=0)
    cam.set_emccd_gain(50)
    assert cam.get_emccd_gain() == 50
    print("[OK] Gain verified: 50")

    # --- Exposure test
    cam.set_exposure_time(0.1)
    actual_exp = cam.get_exposure_time()
    assert abs(actual_exp - 0.1) < 0.01
    print(f"[OK] Exposure time verified: {actual_exp}s")

    # --- ROI and binning
    cam.set_roi(100, 200, 50, 150)
    cam.set_binning(1, 1)
    assert cam.get_roi() == (100, 200, 50, 150)
    assert cam.get_binning() == (1, 1)
    print("[OK] ROI and binning verified.")

    # --- Flip setting test
    cam.set_flip_image_horizontally(True)
    assert cam.get_flip_image_horizontally() is True
    print("[OK] Flip image horizontally enabled.")

    # --- VSS index test
    cam.set_vss_index(1)
    assert cam.get_vss_index() == 1
    print("[OK] VSS index set and verified.")

    # --- Preamp mode test
    cam.set_fixed_pre_amp_mode(4)
    assert cam.get_pre_amp_mode() == 4
    print("[OK] Preamp mode set and verified.")

    # --- Snap image test
    shape = cam.get_frame_shape()
    frame, _ = cam.snap()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == shape
    assert frame.dtype in [np.uint16, np.uint32, np.float64]
    print("[OK] Snap successful with shape:", frame.shape)

    # --- Shutter test
    cam.open_shutter()
    assert cam.cam.get_shutter() == "open"
    cam.close_shutter()
    assert cam.cam.get_shutter() == "closed"
    print("[OK] Shutter open/close tested.")

    # --- Final temp check
    print(f"[INFO] Temperature reading: {cam.cam.get_temperature():.2f} °C")

    cam.close()
    assert not cam.is_opened()
    print("[OK] Camera closed successfully.")

    print("=== ✅ All IxonUltra tests (no cooling) passed ===")

if __name__ == "__main__":
    test_ixon_camera()
