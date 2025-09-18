import queue
import threading
import time

import cv2
from vimba import VimbaFeatureError, PixelFormat

from brillouin_system.devices.cameras.allied.allied_config.allied_config import AlliedConfig, allied_config
from brillouin_system.devices.cameras.allied.dual.base_dual_cameras import BaseDualCameras
from brillouin_system.devices.cameras.allied.single.allied_vision_camera import AlliedVisionCamera

frame_q0 = queue.Queue(maxsize=20)
frame_q1 = queue.Queue(maxsize=20)


def _handler0(cam, frame):
    if frame.get_status() != 0:
        print(f"Cam0: Incomplete frame: {frame.get_status()}!")
    frame_q0.put(frame)
    cam.queue_frame(frame)


def _handler1(cam, frame):
    if frame.get_status() != 0:
        print(f"Cam1: Incomplete frame: {frame.get_status()}!")
    frame_q1.put(frame)
    cam.queue_frame(frame)

class DualAlliedVisionCameras(BaseDualCameras):
    def __init__(self, id0="DEV_000F315BC084", id1="DEV_000F315BDC0C"):
        print("[DualCamera] Initializing two Allied Vision cameras...")

        self.left = AlliedVisionCamera(id=id0)
        self.right = AlliedVisionCamera(id=id1)
        # self.cam0.set_roi(1000, 1000, 200, 200)
        # self.cam1.set_roi(1000, 1000, 200, 200)
        self.left.set_max_roi()
        self.right.set_max_roi()

        self.set_configs(left_cfg=allied_config["left"], right_cfg=allied_config["right"])

        self._setup_snap_mode()
        self._is_streaming = False
        self.start_stream()




    def _setup_snap_mode(self):
        """Configure both cameras for software-triggered snap mode."""
        for cam in [self.left, self.right]:
            cam.set_software_trigger()


    def trigger_both(self):
        self.left.camera.get_feature_by_name("TriggerSoftware").run()
        self.right.camera.get_feature_by_name("TriggerSoftware").run()
        # t1 = threading.Thread(target=lambda: self.cam0.camera.get_feature_by_name("TriggerSoftware").run())
        # t2 = threading.Thread(target=lambda: self.cam1.camera.get_feature_by_name("TriggerSoftware").run())
        # t1.start()
        # t2.start()
        # t1.join()
        # t2.join()

    def start_stream(self):
        """Start streaming once and keep queues ready."""
        self.left.camera.start_streaming(_handler0, buffer_count=10)
        self.right.camera.start_streaming(_handler1, buffer_count=10)
        time.sleep(1)  # Let the queues settle
        self._is_streaming = True

    def stop_stream(self):
        self.left.camera.stop_streaming()
        self.right.camera.stop_streaming()
        self._is_streaming = False

    def clear_queues(self):
        while not frame_q0.empty(): frame_q0.get_nowait()
        while not frame_q1.empty(): frame_q1.get_nowait()

    def snap_once(self, timeout=5.0):
        if not self._is_streaming:
            print("Cameras are not streaming. Start streaming first")
            return None, None

        self.clear_queues()
        self.trigger_both()

        f0 = frame_q0.get(timeout=timeout)
        f1 = frame_q1.get(timeout=timeout)

        return f0, f1

    # def snap_once(self, timeout=2.0):
    #     # # Switch to SingleFrame mode
    #     # self.cam0.set_acquisition_mode("SingleFrame")
    #     # self.cam1.set_acquisition_mode("SingleFrame")
    #     #
    #     # # Trigger both
    #     # self.trigger_both()
    #
    #     # Pull frames directly (blocking)
    #     f0 = self.cam0.snap()
    #     f1 = self.cam1.snap()
    #
    #     if f0.get_status() != 0:
    #         raise RuntimeError("Cam0 returned incomplete frame.")
    #     if f1.get_status() != 0:
    #         raise RuntimeError("Cam1 returned incomplete frame.")
    #
    #     return f0, f1

    def set_configs(self, left_cfg: AlliedConfig | None, right_cfg: AlliedConfig | None):
        """
        Apply configuration objects to both cameras.

        Args:
            left_cfg (AlliedConfig): Configuration for cam0.
            right_cfg (AlliedConfig): Configuration for cam1.
        """
        print("[DualCamera] Applying configs to both cameras...")
        was_streaming = self._is_streaming  # track if we were streaming before

        try:
            if was_streaming:
                self.stop_stream()

            if left_cfg is not None:
                self.left.set_config(left_cfg)
            if right_cfg is not None:
                self.right.set_config(right_cfg)

            print("[DualCamera] Configs applied successfully.")

        except Exception as e:
            print(f"[DualCamera] Failed to apply configs: {e}")
        finally:
            if was_streaming:
                self.start_stream()

    def close(self):
        """Close both cameras cleanly."""
        self.stop_stream()
        self.left.close()
        self.right.close()
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

        # Show the images
        cv2.imshow("Cam 0", img0)
        cv2.imshow("Cam 1", img1)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

        # Show the images
        cv2.imshow("Cam 0", img0)
        cv2.imshow("Cam 1", img1)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Time delta between snaps (based on cam0)
        print(f"Time between first and second snap (cam0): {(t1_0 - t0_0) / 1e6:.3f} ms")
        print(f"Time between first and second snap (cam1): {(t1_1 - t0_1) / 1e6:.3f} ms")

    finally:

        cams.close()

