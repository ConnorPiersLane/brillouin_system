import time
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import gaussian_filter

from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from .andor_dataclasses import AndorExposure, AndorCameraInfo
from .baseCamera import BaseCamera




class DummyCamera(BaseCamera):
    def __init__(self):
        self.exposure_time = 0.3
        self.gain = 1
        self.roi = (0, 160, 0, 20)
        self.binning = (1, 1)
        self.verbose = True
        self._is_streaming = False
        # NEW ATTRIBUTES
        self._flip = False
        self._pre_amp_mode = 16
        self._vss_index = 4
        self._streaming_img_count = 0

        if self.verbose:
            print("[DummyCamera] initialized")

    def get_camera_info(self):
        model = "Simulated - DummyCamera"
        return {
            "model": model,
            "serial": "DUMMY001",
            "roi": self.get_roi(),
            "binning": self.get_binning(),
            "gain": self.get_emccd_gain(),
            "exposure": self.get_exposure_time(),
            "amp_mode": self.get_amp_mode(),
            "preamp_gain": self.get_preamp_gain(),
            "temperature": "off",
            "flip_image_horizontally": self.get_flip_image_horizontally(),
            "advanced_gain_option": False,
            "vss_speed": self.get_vss_index()
        }

    def get_camera_info_dataclass(self) -> AndorCameraInfo:
        return AndorCameraInfo(**self.get_camera_info())

    def set_from_camera_info(self, info: AndorCameraInfo, do_set_temperature: bool = False):

        if self.verbose:
            print("[DummyCamera] Applying settings from AndorCameraInfo...")

        # Flip
        self.set_flip_image_horizontally(info.flip_image_horizontally)

        # ROI + binning
        x_start, x_end, y_start, y_end = info.roi
        hbin, vbin = info.binning
        self.set_roi(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end)
        self.set_binning(hbin=hbin, vbin=vbin)

        # Exposure + gain
        self.set_exposure_time(info.exposure)
        self.set_emccd_gain(int(info.gain))
        self.set_fixed_pre_amp_mode(info.fixed_pre_amp_mode_index)


        # VSS index
        try:
            self.set_vss_index(int(info.vss_speed))
        except Exception:
            if self.verbose:
                print("[DummyCamera] Could not apply vss_speed; keeping existing VSS index.")

        # Temperature (not simulated)
        if do_set_temperature:
            print(f"[DummyCamera] Temperature requested from info: {info.temperature} (dummy camera ignores this)")

        if self.verbose:
            print("[DummyCamera] Camera state restored from AndorCameraInfo.")

    def get_name(self) -> str:
        return "DummyCamera"

    def open_shutter(self):
        print("[DummyCamera] Shutter open")

    def close_shutter(self):
        print("[DummyCamera] Shutter closed")

    def snap(self) -> tuple[np.ndarray, float]:
        time.sleep(self.exposure_time)
        frame = self._generate_plastic_image()
        if self._flip:
            frame = np.fliplr(frame)
        return frame, time.time()

    def _generate_plastic_image(self) -> np.ndarray:
        h, w = self.get_frame_shape()
        image = np.random.normal(loc=150, scale=10, size=(h, w))

        def lorentzian(xx, amp, cen, wid):
            return amp * wid ** 2 / ((xx - cen) ** 2 + wid ** 2)

        x = np.arange(w)
        peak1 = lorentzian(x, amp=1000, cen=w // 2 - 20, wid=4)
        peak2 = lorentzian(x, amp=1000, cen=w // 2 + 20, wid=4)
        spectrum_line = peak1 + peak2 + 200 + np.random.normal(0, 15, size=w)

        band_y = h // 2 + np.random.randint(-2, 2)
        for offset in [-1, 0, 1]:
            image[band_y + offset, :] += spectrum_line

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

    def close(self):
        print("[DummyCamera] Closed.")

    def get_frame_shape(self) -> tuple[int, int]:
        return self.roi[3]-self.roi[2], self.roi[1]-self.roi[0]

    def get_verbose(self) -> bool:
        return self.verbose

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose
        print(f"[DummyCamera] set to self.verbose={self.verbose}")

    def get_preamp_gain(self) -> int:
        return 1

    def get_amp_mode(self) -> str:
        return f"DummyAmpMode(preamp_mode={self._pre_amp_mode})"

    # NEW: Flip image horizontally
    def set_flip_image_horizontally(self, flip: bool):
        self._flip = flip
        if self.verbose:
            print(f"[DummyCamera] Flip image horizontally set to {flip}")

    def get_flip_image_horizontally(self) -> bool:
        return self._flip

    # NEW: Preamp mode
    def set_fixed_pre_amp_mode(self, index: int):
        self._pre_amp_mode = index
        if self.verbose:
            print(f"[DummyCamera] Preamp mode set to index {index}")


    def get_fixed_pre_amp_mode(self) -> int:
        return self._pre_amp_mode

    def get_pre_amp_mode(self) -> int:
        return self._pre_amp_mode

    # NEW: VSS index
    def set_vss_index(self, index: int):
        self._vss_index = index
        if self.verbose:
            print(f"[DummyCamera] VSS index set to {index}")

    def get_vss_index(self) -> int:
        return self._vss_index

    def set_from_config_file(self, config: AndorConfig) -> None:
        if self.verbose:
            print("[DummyCamera] Applying settings from config...")


        self.set_verbose(config.verbose)
        self.set_flip_image_horizontally(config.flip_image_horizontally)

        self.set_roi(
            x_start=config.x_start,
            x_end=config.x_end,
            y_start=config.y_start,
            y_end=config.y_end
        )

        self.set_binning(
            hbin=config.hbin,
            vbin=config.vbin
        )

        self.set_fixed_pre_amp_mode(config.pre_amp_mode)
        self.set_vss_index(config.vss_index)

        print(f'Temperature is {config.temperature}')


        if self.verbose:
            print("[IxonUltra] Configuration applied.")


    def get_exposure_dataclass(self) -> AndorExposure:
        return AndorExposure(
            exposure_time_s=self.get_exposure_time(),
            emccd_gain=self.get_emccd_gain()
        )

    def start_streaming(self, buffer_size: int = 200):
        self._is_streaming = True
        print(f"Started Streaming: buffer {buffer_size}")

    def stop_streaming(self):
        self._is_streaming = False
        print("Ended Streaming")

    def get_newest_streaming_image(self):
        """
        only when streaming
        Return the newest frame available (non-blocking).
        Returns None if no *new* frame since last call.
        """
        return self.snap()


    @contextmanager
    def streaming(self):
        """
        Safe streaming context manager.
        """
        already_streaming = self._is_streaming
        if not already_streaming:
            self.start_streaming()

        try:
            yield self
        finally:
            if not already_streaming and self._is_streaming:
                self.stop_streaming()