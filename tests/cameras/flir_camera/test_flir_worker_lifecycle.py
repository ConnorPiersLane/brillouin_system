import sys
import time
from PyQt5.QtWidgets import QApplication
  # Make sure this matches your filename
import numpy as np

from brillouin_system.devices.cameras.flir.flir_cam import FLIRCamera
from brillouin_system.devices.cameras.flir.flir_worker import FlirWorker


def dummy_frame_handler(frame: np.ndarray):
    print(f"[TEST] Got stream frame of shape {frame.shape}")

def test_flir_worker_lifecycle():
    app = QApplication(sys.argv)

    print("\n[TEST] Initializing camera and worker...")
    cam = FLIRCamera()
    worker = FlirWorker(cam, fps=5)

    # 1. Start streaming
    print("\n[TEST] Starting stream...")
    worker.start_stream(dummy_frame_handler)
    print(worker.cam.min_max_exposure_time())
    print(worker.cam.min_max_gain())
    print(worker.cam.min_max_gamma())
    worker.update_exposure_gain_gamma(gamma=1)
    time.sleep(3)

    # 2. Stop stream
    print("\n[TEST] Stopping stream...")
    worker.stop_stream()
    worker._thread.quit()
    worker._thread.wait()

    time.sleep(1)

    # 3. Enable snap mode and capture one frame
    print("\n[TEST] Enabling snap mode and capturing one frame...")
    worker.enable_software_snap()
    snap = worker.software_snap()
    if snap is not None:
        print(f"[TEST] Snap frame shape: {snap.shape}")
    else:
        print("[TEST] Snap failed.")

    # 4. Go back to streaming
    print("\n[TEST] Restarting stream...")
    worker.start_stream(dummy_frame_handler)
    time.sleep(2)

    # 5. Final shutdown
    print("\n[TEST] Shutting down...")
    worker.shutdown()

    print("\n[TEST] Done.")
    sys.exit(0)

if __name__ == "__main__":
    test_flir_worker_lifecycle()
