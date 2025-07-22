import numpy as np
import cv2


class DummyFLIRCamera:
    def __init__(self, index=0, width=2000, height=2000):
        """
        A dummy FLIR camera for development without hardware.
        Simulates acquisition by returning black images.
        """
        self._max_width = 3208
        self._max_height = 2200
        self._roi = (0, 0, width, height)
        self._gain = 0.0
        self._exposure = 20000.0
        self._pixel_format = 'Mono16'
        self._is_software_stream = False
        self._is_single_frame_mode = False

        print(f"[DummyFLIR] Initialized with ROI: {self._roi}")

    def configure_camera(self):
        print("[DummyFLIR] configure_camera() called — no-op.")

    def get_camera_info(self):
        info = {
            "model": "DummyFLIR",
            "serial": "00000000",
            "sensor_size": self.get_sensor_size(),
            "roi": self.get_roi_native(),
            "gain": self.get_gain(),
            "exposure": self.get_exposure_time(),
            "pixel_format": self.get_pixel_format()
        }
        print(f"[DummyFLIR] get_camera_info() -> {info}")
        return info

    def set_resolution(self, width, height):
        self._roi = (0, 0, min(width, self._max_width), min(height, self._max_height))
        print(f"[DummyFLIR] Resolution set to {self._roi[2]} x {self._roi[3]}")

    def get_resolution(self):
        res = self._roi[2], self._roi[3]
        print(f"[DummyFLIR] get_resolution() -> {res}")
        return res

    def get_sensor_size(self):
        size = self._max_width, self._max_height
        print(f"[DummyFLIR] get_sensor_size() -> {size}")
        return size

    def set_max_roi(self):
        self._roi = (0, 0, self._max_width, self._max_height)
        print(f"[DummyFLIR] Max ROI set: {self._roi}")

    def set_roi_native(self, offset_x, offset_y, width, height):
        width = min(width, self._max_width)
        height = min(height, self._max_height)
        self._roi = (offset_x, offset_y, width, height)
        print(f"[DummyFLIR] ROI set to {self._roi}")

    def get_roi_native(self):
        print(f"[DummyFLIR] get_roi_native() -> {self._roi}")
        return self._roi

    def set_gain(self, value_dB):
        self._gain = value_dB
        print(f"[DummyFLIR] Gain set to {value_dB} dB")

    def get_gain(self):
        print(f"[DummyFLIR] get_gain() -> {self._gain}")
        return self._gain

    def set_gamma(self, value):
        print(f"[DummyFLIR] Gamma set to {value}")

    def get_gamma(self):
        print(f"[DummyFLIR] get_gamma() -> None (not simulated)")
        return None

    def set_exposure_time(self, value):
        self._exposure = value
        print(f"[DummyFLIR] Exposure time set to {value} µs")

    def get_exposure_time(self):
        print(f"[DummyFLIR] get_exposure_time() -> {self._exposure}")
        return self._exposure

    def min_max_gain(self):
        return 0, 100

    def min_max_exposure_time(self):
        return 1, 100000

    def min_max_gamma(self):
        return 0, 40


    def get_pixel_format(self):
        print(f"[DummyFLIR] get_pixel_format() -> {self._pixel_format}")
        return self._pixel_format

    def set_pixel_format(self, format_str):
        self._pixel_format = format_str
        print(f"[DummyFLIR] Pixel format set to {format_str}")

    def start_single_frame_mode(self):
        self._is_single_frame_mode = True
        self._is_software_stream = False
        print("[DummyFLIR] Single frame mode enabled")

    def acquire_image(self, timeout=1000):
        width, height = self.get_resolution()
        dtype = np.uint16 if self._pixel_format == 'Mono16' else np.uint8
        print(f"[DummyFLIR] Acquiring image ({width}x{height}, {dtype})")
        return np.zeros((height, width), dtype=dtype)

    def start_software_stream(self):
        self._is_software_stream = True
        self._is_single_frame_mode = False
        print("[DummyFLIR] Software streaming started")

    def software_snap_while_stream(self, timeout=1000):
        print("[DummyFLIR] software_snap_while_stream() called")
        return self.acquire_image(timeout)

    def end_software_stream(self):
        self._is_software_stream = False
        print("[DummyFLIR] Software streaming stopped")

    def shutdown(self):
        print("[DummyFLIR] Shutdown called")

    def get_available_pixel_formats(self):
        formats = ['Mono8', 'Mono16', 'Mono10Packed', 'Mono12Packed', 'Mono10p', 'Mono12p']
        print(f"[DummyFLIR] get_available_pixel_formats() -> {formats}")
        return formats

    def __del__(self):
        print("[DummyFLIR] __del__ called")
        self.shutdown()
