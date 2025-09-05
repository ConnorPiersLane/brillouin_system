import numpy as np


from brillouin_system.devices.cameras.allied.dual.base_dual_cameras import BaseDualCameras


class DummyDualCameras(BaseDualCameras):
    def __init__(self, id0="DUMMY0", id1="DUMMY1"):
        super().__init__(id0, id1)
        self.initialize_cameras()

    def initialize_cameras(self):
        print("[DummyCamera] Initializing dummy cameras...")

    def trigger_both(self):
        """No-op for dummy implementation (snap_once generates frames)."""
        pass

    def start_stream(self):
        print("[DummyCamera] Start stream (no-op).")

    def stop_stream(self):
        print("[DummyCamera] Stop stream (no-op).")

    def close(self):
        print("[DummyCamera] Cameras closed (no-op).")

    def snap_once(self, timeout=1.0):
        """Generate two random salt-and-pepper noise frames."""
        h, w = 480, 640
        noise_density = 0.05  # 5% pixels are noise

        def salt_pepper_noise():
            img = np.zeros((h, w), dtype=np.uint8) + 127  # mid-gray base
            # random mask for salt
            num_salt = int(noise_density * h * w / 2)
            coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
            img[coords] = 255
            # random mask for pepper
            num_pepper = int(noise_density * h * w / 2)
            coords = (np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper))
            img[coords] = 0
            return img

        f0 = salt_pepper_noise()
        f1 = salt_pepper_noise()
        return f0, f1
