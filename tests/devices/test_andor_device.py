import threading
import numpy as np
import time

from brillouinDAQ.devices.AndorDevice import AndorDevice

# Initialize a threading event for stopping the test
stop_event = threading.Event()


# Mock an application class with a lock
class MockApp:
    def __init__(self):
        self.andor_lock = threading.Lock()


# Create an instance of AndorDevice using the real hardware
app = MockApp()
andor_device = AndorDevice(stop_event, app)


def test_camera_initialization():
    """ Test if the Andor camera initializes correctly. """
    try:
        print("[TEST] Initializing Andor Camera...")
        andor_device.cam.Initialize()
        print("[PASS] Camera initialized successfully.")
    except Exception as e:
        print(f"[FAIL] Camera initialization failed: {e}")


def test_set_and_get_exposure():
    """ Test setting and getting exposure time. """
    try:
        test_exposure = 0.1  # seconds
        print(f"[TEST] Setting exposure time to {test_exposure} sec...")
        andor_device.setExposure(test_exposure)
        retrieved_exposure = andor_device.getExposure()
        assert abs(retrieved_exposure - test_exposure) < 0.01, "Exposure mismatch!"
        print("[PASS] Exposure time set and retrieved correctly.")
    except Exception as e:
        print(f"[FAIL] Setting exposure time failed: {e}")


def test_data_acquisition():
    """ Test if the camera can capture an image. """
    try:
        print("[TEST] Acquiring image from Andor camera...")
        image_data = andor_device.getData()
        assert image_data is not None, "Image data is None!"
        assert isinstance(image_data, np.ndarray), "Image data is not a NumPy array!"
        print(f"[PASS] Image acquired successfully. Shape: {image_data.shape}")
    except Exception as e:
        print(f"[FAIL] Image acquisition failed: {e}")


def test_background_subtraction():
    """ Test the background subtraction feature, ensuring it follows the actual application behavior. """
    try:
        print("[TEST] Performing background subtraction...")

        # Start background subtraction
        andor_device.startBGsubtraction()
        time.sleep(1)  # Give some time for background acquisition

        # Check that subtraction has started
        if andor_device.checkBGsubtraction():
            print("[PASS] Background subtraction triggered successfully.")
        else:
            print("[FAIL] Background subtraction not triggered.")

        # Wait for background subtraction to complete (just like run_macro.py does)
        while andor_device.checkBGsubtraction():
            print("[INFO] Waiting for background subtraction to finish...")
            _ = andor_device.getData()  # Acquire an image to reset triggerBG
            time.sleep(0.5)  # Small delay to prevent CPU overload

        print("[PASS] Background subtraction completed.")

        # Stop background subtraction
        andor_device.stopBGsubtraction()

        # Verify background subtraction is truly off
        if not andor_device.checkBGsubtraction():
            print("[PASS] Background subtraction stopped successfully.")
        else:
            print("[FAIL] Background subtraction still active after stopping.")

    except Exception as e:
        print(f"[FAIL] Background subtraction test failed: {e}")
def run_all_tests():
    """ Run all tests in sequence. """
    test_camera_initialization()
    test_set_and_get_exposure()
    test_data_acquisition()
    test_background_subtraction()
    print("\nAll tests completed.")


# Run the tests
if __name__ == "__main__":
    run_all_tests()
