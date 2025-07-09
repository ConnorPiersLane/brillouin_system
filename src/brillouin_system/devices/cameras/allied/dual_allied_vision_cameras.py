import queue
import threading
import time
from allied_vision_camera import AlliedVisionCamera

frame_q0 = queue.Queue()
frame_q1 = queue.Queue()

def _handler0(cam, frame):
    frame_q0.put(frame)
    cam.queue_frame(frame)


def _handler1(cam, frame):
    frame_q1.put(frame)
    cam.queue_frame(frame)

class DualAlliedVisionCameras:
    def __init__(self, id0="DEV_000F315BC084", id1="DEV_000F315BDC0C"):
        print("[DualCamera] Initializing two Allied Vision cameras...")

        self.cam0 = AlliedVisionCamera(id=id0)
        self.cam1 = AlliedVisionCamera(id=id1)

        self._setup_snap_mode()

    def _setup_snap_mode(self):
        """Configure both cameras for software-triggered snap mode."""
        for cam in [self.cam0, self.cam1]:
            cam.set_software_trigger()


    def trigger_both(self):
        t1 = threading.Thread(target=lambda: self.cam0.camera.get_feature_by_name("TriggerSoftware").run())
        t2 = threading.Thread(target=lambda: self.cam1.camera.get_feature_by_name("TriggerSoftware").run())
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def start_stream(self):
        """Start streaming once and keep queues ready."""
        self.cam0.camera.start_streaming(_handler0)
        self.cam1.camera.start_streaming(_handler1)
        # time.sleep(0.2)  # Let the queues settle

    def stop_stream(self):
        self.cam0.camera.stop_streaming()
        self.cam1.camera.stop_streaming()

    def clear_queues(self):
        while not frame_q0.empty(): frame_q0.get_nowait()
        while not frame_q1.empty(): frame_q1.get_nowait()

    def snap_once(self, timeout=5.0):
        self.start_stream()
        self.clear_queues()
        self.trigger_both()

        f0 = frame_q0.get(timeout=timeout)
        f1 = frame_q1.get(timeout=timeout)

        self.stop_stream()
        return f0, f1

    def close(self):
        """Close both cameras cleanly."""
        self.cam0.close()
        self.cam1.close()
        print("[DualCamera] Cameras closed.")


if __name__ == "__main__":
    cams = DualAlliedVisionCameras()

    try:
        # First snap
        f0, f1 = cams.snap_once()
        t0_0 = f0.get_timestamp()
        t0_1 = f1.get_timestamp()
        img0 = f0.as_numpy_ndarray()
        img1 = f1.as_numpy_ndarray()
        print("First Snap:")
        print("  Cam0 Frame shape:", img0.shape)
        print("  Cam1 Frame shape:", img1.shape)
        print(t0_0)
        print(t0_1)
        print(f"  Time delta between cams: {(abs(t0_0 - t0_1)) / 1e6:.3f} ms")

        # Second snap
        f0, f1 = cams.snap_once()
        t1_0 = f0.get_timestamp()
        t1_1 = f1.get_timestamp()
        img0 = f0.as_numpy_ndarray()
        img1 = f1.as_numpy_ndarray()
        print("Second Snap:")
        print("  Cam0 Frame shape:", img0.shape)
        print("  Cam1 Frame shape:", img1.shape)
        print(t1_0)
        print(t1_1)
        print(f"  Time delta between cams: {(abs(t1_0 - t1_1)) / 1e6:.3f} ms")

        # Time delta between snaps (based on cam0)
        print(f"Time between first and second snap (cam0): {(t1_0 - t0_0) / 1e6:.3f} ms")
        print(f"Time between first and second snap (cam1): {(t1_1 - t0_1) / 1e6:.3f} ms")

    finally:

        cams.close()

