import time
import numpy as np
from scipy.ndimage import gaussian_filter
from .baseCamera import BaseCamera

class DummyCamera(BaseCamera):
    def __init__(self):
        # Default settings
        self.exposure_time = 0.3
        self.gain = 1
        self.roi = (0, 160, 0, 20)
        self.binning = (1, 1)
        self.frame_shape = (20, 160)  # height, width (shorter height, wider image)
        self.verbose = True
        if self.verbose:
            print("[DummyCamera] initialized")

    def get_name(self) -> str:
        return "DummyCamera"

    def open_shutter(self):
        print("[DummyCamera] Shutter open")

    def close_shutter(self):
        print("[DummyCamera] Shutter closed")

    def snap(self) -> np.ndarray:
        time.sleep(self.exposure_time)
        frame = self._generate_plastic_image()
        return frame

    def _generate_plastic_image(self) -> np.ndarray:
        h, w = self.frame_shape
        image = np.random.normal(loc=150, scale=10, size=(h, w))

        # --- Create synthetic Brillouin spectrum (2 Lorentzian peaks)
        def lorentzian(xx, amp, cen, wid):
            return amp * wid ** 2 / ((xx - cen) ** 2 + wid ** 2)

        x = np.arange(w)
        peak1 = lorentzian(x, amp=1000, cen=w // 2 - 30, wid=4)
        peak2 = lorentzian(x, amp=1000, cen=w // 2 + 30, wid=4)
        spectrum_line = peak1 + peak2 + 200 + np.random.normal(0, 15, size=w)

        # --- Inject this synthetic line into a horizontal band
        band_y = h // 2 + np.random.randint(-2, 2)
        for offset in [-1, 0, 1]:  # 3-row band
            image[band_y + offset, :] += spectrum_line

        # Smooth the full image for realism
        image = gaussian_filter(image, sigma=1.2)
        return np.clip(image, 0, 65535).astype(np.uint16)

    def set_exposure_time(self, seconds: float):
        self.exposure_time = seconds

    def get_exposure_time(self) -> float:
        return self.exposure_time

    def set_emccd_gain(self, gain: float | int):
        self.gain = gain

    def get_emccd_gain(self) -> float | int:
        return self.gain

    def set_roi(self, x_start: int, x_end: int, y_start: int, y_end: int):
        self.roi = (x_start, x_end, y_start, y_end)

    def get_roi(self) -> tuple[int, int, int, int]:
        return self.roi

    def set_binning(self, hbin: int, vbin: int):
        self.binning = (hbin, vbin)

    def get_binning(self) -> tuple[int, int]:
        return self.binning

    def is_opened(self) -> bool:
        return True

    def get_preamp_gain(self) -> int:
        """Preamp gain (e⁻/count)"""
        return 1.0

    def get_amp_mode(self) -> tuple:
        """

        """
        return "Test Amp Mode: (channel, oamp, hsspeed, preamp)"

    def close(self):
        pass

    def get_frame_shape(self) -> tuple[int, int]:
        return self.frame_shape

    def get_verbose(self) -> bool:
        return self.verbose

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose
        print(f"[DummyCamera] set to self.verbose={self.verbose}")
