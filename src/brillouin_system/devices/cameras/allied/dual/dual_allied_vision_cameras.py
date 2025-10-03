import queue
import threading
import time

import cv2
import numpy as np
from vimba import VimbaFeatureError, PixelFormat

from brillouin_system.devices.cameras.allied.allied_config.allied_config import AlliedConfig, allied_config
from brillouin_system.devices.cameras.allied.dual.base_dual_cameras import BaseDualCameras
from brillouin_system.devices.cameras.allied.single.allied_vision_camera import AlliedVisionCamera

# Queues for the latest frames
frame_q0 = queue.Queue(maxsize=20)
frame_q1 = queue.Queue(maxsize=20)


def clear_queues():
    while not frame_q0.empty():
        frame_q0.get_nowait()
    while not frame_q1.empty():
        frame_q1.get_nowait()


def _push(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()  # drop oldest
        except queue.Empty:
            pass
        q.put_nowait(item)


def _to_mono2d(a):
    """Ensure a 2D (H, W) array for mono images."""
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]   # drop the singleton channel
    return a


# --- Robust streaming handlers: always requeue in finally ---

def _handler0(cam, frame):
    try:
        if frame.get_status() != 0:
            print(f"Cam0: Incomplete frame: {frame.get_status()}!")
        img = frame.as_numpy_ndarray().copy()  # deep copy to detach from Vimba buffer
        _push(frame_q0, img)
    except Exception as e:
        print(f"Cam0 handler error: {e}")
    finally:
        cam.queue_frame(frame)


def _handler1(cam, frame):
    try:
        if frame.get_status() != 0:
            print(f"Cam1: Incomplete frame: {frame.get_status()}!")
        else:
            img = frame.as_numpy_ndarray().copy()
            _push(frame_q1, img)
    except Exception as e:
        print(f"Cam1 handler error: {e}")
    finally:
        cam.queue_frame(frame)


# --- Helpers to tune GVSP pacing / limits ---

def _set_inter_packet_delay(camera, delay_ticks: int | None):
    """Best-effort set of inter-packet delay; name varies by model.
    A small delay (e.g., 800–1200) helps de-phase two cameras sharing one link.
    """
    if delay_ticks is None:
        return
    for name in ("GevSCPD", "GevSCPPacketDelay", "GVSPPacketDelay"):
        try:
            camera.get_feature_by_name(name).set(int(delay_ticks))
            return
        except Exception:
            continue


def _enable_ptp(camera, on: bool = True):
    try:
        camera.get_feature_by_name("GevIEEE1588").set("On" if on else "Off")
    except Exception:
        pass


class DualAlliedVisionCameras(BaseDualCameras):
    def __init__(
        self,
        id0: str = "DEV_000F315BC084",
        id1: str = "DEV_000F315BDC0C",
        *,
        throughput_MBps_per_cam: float = 55.0,  # ~110 MB/s total budget across both on 1GbE
        packet_delays: tuple[int | None, int | None] = (800, 1200),
        enable_ptp: bool = False,
        pixel_format = PixelFormat.Mono8,
    ):
        print("[DualCamera] Initializing two Allied Vision cameras.")

        # NOTE: AlliedVisionCamera.init currently interprets its `limit_mbps` as MB/s.
        # Passing 55 gives ~55 MB/s per camera.
        self.left = AlliedVisionCamera(id=id0, pixel_format='Mono8')
        self.right = AlliedVisionCamera(id=id1, pixel_format='Mono8')

        # Apply per-camera transport limits & pacing
        self._set_transport(self.left.camera, throughput_MBps_per_cam, packet_delays[0])
        self._set_transport(self.right.camera, throughput_MBps_per_cam, packet_delays[1])

        if enable_ptp:
            _enable_ptp(self.left.camera, True)
            _enable_ptp(self.right.camera, True)

        self._is_streaming = False

        # Apply camera-specific configs if provided externally
        left_cfg = allied_config["left"].get()
        right_cfg = allied_config["right"].get()
        self.set_configs(left_cfg=left_cfg, right_cfg=right_cfg)

        # Ensure pixel format (Mono8 → smallest payload)
        try:
            self.left.camera.set_pixel_format(pixel_format)
            self.right.camera.set_pixel_format(pixel_format)
        except VimbaFeatureError as e:
            print(f"[DualCamera] Pixel format not applied: {e}")

        # Use software-trigger snap mode by default
        self._setup_snap_mode()

        # Start streaming now; queues will hold latest frames
        self.start_stream()

    def _set_transport(self, cam, mbps_per_cam: float, delay_ticks: int | None):
        # Bytes per second cap: AlliedVisionCamera.init uses bytes/s features
        try:
            for name in ("DeviceLinkThroughputLimitMode",):
                cam.get_feature_by_name(name).set("On")
        except Exception:
            pass

        bytes_per_sec = int(mbps_per_cam * 1_000_000)  # interpret as MB/s
        for name in ("DeviceLinkThroughputLimit", "StreamBytesPerSecond"):
            try:
                cam.get_feature_by_name(name).set(bytes_per_sec)
            except Exception:
                pass

        # Maximize packet size (driver/switch must allow jumbo frames for full effect)
        for name in ("GVSPAdjustPacketSize",):
            try:
                cam.get_feature_by_name(name).run()
            except Exception:
                pass

        # Optional pacing to reduce microbursts across two streams
        _set_inter_packet_delay(cam, delay_ticks)

        # Favor fresh frames when host stalls
        try:
            cam.get_feature_by_name("StreamBufferHandlingMode").set("NewestOnly")
        except Exception:
            pass
        for name in ("StreamBufferCountMode",):
            try:
                cam.get_feature_by_name(name).set("Manual")
                cam.get_feature_by_name("StreamBufferCountManual").set(40)
            except Exception:
                pass

    def _setup_snap_mode(self):
        """Configure both cameras for software-triggered snap mode."""
        for cam in [self.left, self.right]:
            cam.set_software_trigger()

    def trigger_both(self):
        # Fire software triggers concurrently for tighter sync
        t1 = threading.Thread(target=lambda: self.left.camera.get_feature_by_name("TriggerSoftware").run())
        t2 = threading.Thread(target=lambda: self.right.camera.get_feature_by_name("TriggerSoftware").run())
        t1.start(); t2.start(); t1.join(); t2.join()

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

    def snap_once(self, timeout: float = 5.0):
        """
        Acquire a single synchronized frame from both cameras (software trigger).
        """
        if not self._is_streaming:
            print("Cameras are not streaming. Start streaming first")
            return None, None

        clear_queues()
        self.trigger_both()

        f0 = frame_q0.get(timeout=timeout)
        f1 = frame_q1.get(timeout=timeout)

        f0 = _to_mono2d(f0)
        f1 = _to_mono2d(f1)
        return f0, f1

    def set_configs(self, left_cfg: AlliedConfig | None, right_cfg: AlliedConfig | None):
        """Apply configuration objects to both cameras."""
        print("[DualCamera] Applying configs to both cameras...")
        was_streaming = self._is_streaming
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
    cams = DualAlliedVisionCameras(
        throughput_MBps_per_cam=55.0,      # ~13 fps each @ 2048x2048 Mono8 on 1GbE
        packet_delays=(800, 1200),         # de-phase streams
        enable_ptp=False,                  # set True if you want shared timebase
    )

    try:
        # First snap
        f0, f1 = cams.snap_once()
        print("First Snap:")
        print("  Cam0 Frame shape:", f0.shape)
        print("  Cam1 Frame shape:", f1.shape)
        cv2.imshow("Cam 0", f0)
        cv2.imshow("Cam 1", f1)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Second snap
        f0, f1 = cams.snap_once()
        print("Second Snap:")
        print("  Cam0 Frame shape:", f0.shape)
        print("  Cam1 Frame shape:", f1.shape)
        cv2.imshow("Cam 0", f0)
        cv2.imshow("Cam 1", f1)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        cams.close()
