import queue
import threading
import time

import cv2
import numpy as np
from vimba import VimbaFeatureError, PixelFormat

from brillouin_system.devices.cameras.allied.allied_config.allied_config import AlliedConfig, allied_config
from brillouin_system.devices.cameras.allied.dual.base_dual_cameras import BaseDualCameras
from brillouin_system.devices.cameras.allied.single.allied_vision_camera import AlliedVisionCamera

frame_q0 = queue.Queue(maxsize=20)
frame_q1 = queue.Queue(maxsize=20)

def _push(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()  # drop oldest
        except queue.Empty:
            pass
        q.put_nowait(item)


# Put (timestamp, image_copy) into the queues instead of the Frame object.
def _handler0(cam, frame):
    status = frame.get_status()
    if status != 0:
        print(f"Cam0: Incomplete frame: {frame.get_status()}!")
    ts = frame.get_timestamp()
    img = frame.as_numpy_ndarray().copy()  # deep copy to detach from Vimba buffer
    _push(frame_q0, (status, ts, img))
    cam.queue_frame(frame)

def _handler1(cam, frame):
    status = frame.get_status()
    if status != 0:
        print(f"Cam1: Incomplete frame: {frame.get_status()}!")
    ts = frame.get_timestamp()
    img = frame.as_numpy_ndarray().copy()
    _push(frame_q1, (status, ts, img))
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
        self._is_streaming = False

        left_cfg = allied_config["left"].get()
        right_cfg = allied_config["right"].get()
        self.set_configs(left_cfg=left_cfg, right_cfg=right_cfg)

        try:
            self.left.camera.set_pixel_format(PixelFormat.Mono8)
            self.right.camera.set_pixel_format(PixelFormat.Mono8)
        except VimbaFeatureError as e:
            print(f"[DualCamera] Pixel format not applied: {e}")

        self._setup_snap_mode()

        self.start_stream()




    def _setup_snap_mode(self):
        """Configure both cameras for software-triggered snap mode."""
        for cam in [self.left, self.right]:
            cam.set_software_trigger()


    def trigger_both(self):
        # self.left.camera.get_feature_by_name("TriggerSoftware").run()
        # self.right.camera.get_feature_by_name("TriggerSoftware").run()
        t1 = threading.Thread(target=lambda: self.left.camera.get_feature_by_name("TriggerSoftware").run())
        t2 = threading.Thread(target=lambda: self.right.camera.get_feature_by_name("TriggerSoftware").run())
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def start_stream(self):
        """Start streaming once and keep queues ready."""
        self.left.camera.start_streaming(_handler0, buffer_count=40)
        self.right.camera.start_streaming(_handler1, buffer_count=40)
        time.sleep(2)  # Let the queues settle
        self._is_streaming = True

    def stop_stream(self):
        self.left.camera.stop_streaming()
        self.right.camera.stop_streaming()
        self._is_streaming = False

    def clear_queues(self):
        while not frame_q0.empty(): frame_q0.get_nowait()
        while not frame_q1.empty(): frame_q1.get_nowait()

    def snap_once(self, timeout=5.0, return_info=False) -> tuple:
        """
        Acquire a single synchronized frame from both cameras.

        This method clears any old frames in the internal queues, triggers both cameras
        via software trigger, and then blocks until a new frame is received from each.

        Args:
            timeout (float, optional): Maximum time (in seconds) to wait for each frame
                before raising `queue.Empty`. Defaults to 5.0.

        Returns:
            tuple[tuple[int, int, np.ndarray], tuple[int, int, np.ndarray]] | tuple[None, None]:
                On success, returns a pair of tuples, one for each camera (left, right):

                - `status` (int): The Vimba frame status code (0 = complete, nonzero = incomplete).
                - `timestamp` (int): Camera hardware timestamp of the frame (nanoseconds).
                - `frame` (np.ndarray): The image data as a NumPy array.

                Example return structure:
                if return_info = True
                    f0, f1, ts0, ts1, status0, status1
                else:
                    f0, f1,

                If streaming is not active, returns `(None, None)` and prints a warning.

        Raises:
            queue.Empty: If no frame is received from either camera within `timeout`.

        Notes:
            - Always call `start_stream()` before using this method.
            - Frames are copied into the queue by the streaming callback; this call
              retrieves the *next* available pair of frames after triggering.
            - Synchronization accuracy is limited by software triggering and host
              scheduling jitter. For tighter sync, use hardware triggers or PTP.
        """
        if not self._is_streaming:
            print("Cameras are not streaming. Start streaming first")
            return None, None

        self.clear_queues()
        self.trigger_both()

        (status0, ts0, f0) = frame_q0.get(timeout=timeout)
        (status1, ts1, f1) = frame_q1.get(timeout=timeout)
        if return_info:
            return f0, f1, ts0, ts1, status0, status1
        else:
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
        f0, f1, ts0, ts1, status0, status1 = cams.snap_once(return_info=True)
        print("First Snap:")
        print("  Cam0 Frame shape:", f0.shape, " Status:", status0, " Timestamp:", ts0)
        print("  Cam1 Frame shape:", f1.shape, " Status:", status1, " Timestamp:", ts1)
        print(f"  Time delta between cams: {(abs(ts0 - ts1)) / 1e6:.3f} ms")

        # Show the images
        cv2.imshow("Cam 0", f0)
        cv2.imshow("Cam 1", f1)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Second snap
        f0, f1, ts0b, ts1b, status0b, status1b = cams.snap_once(return_info=True)
        print("Second Snap:")
        print("  Cam0 Frame shape:", f0.shape, " Status:", status0b, " Timestamp:", ts0b)
        print("  Cam1 Frame shape:", f1.shape, " Status:", status1b, " Timestamp:", ts1b)
        print(f"  Time delta between cams: {(abs(ts0b - ts1b)) / 1e6:.3f} ms")

        # Show the images
        cv2.imshow("Cam 0", f0)
        cv2.imshow("Cam 1", f1)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Time delta between snaps (based on cam0 and cam1)
        print(f"Time between first and second snap (cam0): {(ts0b - ts0) / 1e6:.3f} ms")
        print(f"Time between first and second snap (cam1): {(ts1b - ts1) / 1e6:.3f} ms")

    finally:
        cams.close()


