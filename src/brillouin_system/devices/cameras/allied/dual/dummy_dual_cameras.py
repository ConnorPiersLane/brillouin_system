import numpy as np
from dataclasses import asdict

from brillouin_system.devices.cameras.allied.allied_config.allied_config import AlliedConfig, allied_config
from brillouin_system.devices.cameras.allied.dual.base_dual_cameras import BaseDualCameras


class DummyDualCameras(BaseDualCameras):
    """
    Dummy stereo camera that mirrors DualAlliedVisionCameras:
      - Applies configs in __init__
      - start_stream() in __init__ sets _is_streaming = True
      - snap_once() returns two np.uint8 frames whose shapes come from configs
    """

    def __init__(self, id0="DUMMY0", id1="DUMMY1"):
        print("[DummyCamera] Initializing dummy cameras...")
        self.left_id = id0
        self.right_id = id1

        # Default shapes (will be overwritten by configs below)
        self._left_shape = (480, 640)    # (H, W)
        self._right_shape = (480, 640)   # (H, W)
        self._dtype = np.uint8
        self._is_streaming = False

        # Try to load "left/right" first; fall back to known device IDs if needed
        left_cfg = allied_config.get("left")
        right_cfg = allied_config.get("right")



        # Apply at init (mirrors your real class behavior)
        self.set_configs(left_cfg, right_cfg)

        # Start "streaming" like the real implementation
        self.start_stream()

    # ---- API mirror of the real class ----

    def start_stream(self):
        print("[DummyCamera] Start stream (flag only).")
        self._is_streaming = True

    def stop_stream(self):
        print("[DummyCamera] Stop stream (flag only).")
        self._is_streaming = False

    def trigger_both(self):
        """No-op for dummy implementation (snap_once generates frames on demand)."""
        pass

    def clear_queues(self):
        """No-op: kept for API symmetry with real class."""
        pass

    def set_configs(self, left_cfg: AlliedConfig | None, right_cfg: AlliedConfig | None):
        """
        Update internal frame sizes based on configs (width/height).
        Mimics real behavior: stop stream while applying, then restart if it was running.
        """
        print("[DummyCamera] Applying configs...")
        was_streaming = self._is_streaming
        try:
            if was_streaming:
                self.stop_stream()

            if left_cfg is not None:
                # Height/Width from config define frame shape
                self._left_shape = (int(left_cfg.height), int(left_cfg.width))
                print(f"[DummyCamera] Left shape set to {self._left_shape}")
            if right_cfg is not None:
                self._right_shape = (int(right_cfg.height), int(right_cfg.width))
                print(f"[DummyCamera] Right shape set to {self._right_shape}")
            print("[DummyCamera] Configs applied successfully.")
        except Exception as e:
            print(f"[DummyCamera] Failed to apply configs: {e}")
        finally:
            if was_streaming:
                self.start_stream()

    def snap_once(self, timeout=1.0):
        """
        Generate two random frames (uint8) using the configured shapes.
        Mirrors the real class behavior of requiring streaming to be active.
        """
        if not self._is_streaming:
            print("Cameras are not streaming. Start streaming first")
            return None, None

        hL, wL = self._left_shape
        hR, wR = self._right_shape

        # Mid-gray with salt & pepper noise (like your previous dummy)
        def salt_pepper(h, w, density=0.05):
            img = np.full((h, w), 127, dtype=self._dtype)
            num = int(density * h * w / 2)
            if num > 0:
                ys = np.random.randint(0, h, num)
                xs = np.random.randint(0, w, num)
                img[ys, xs] = 255
                ys = np.random.randint(0, h, num)
                xs = np.random.randint(0, w, num)
                img[ys, xs] = 0
            return img

        left = salt_pepper(hL, wL)
        right = salt_pepper(hR, wR)
        return left, right

    def close(self):
        print("[DummyCamera] Cameras closed (no-op).")
