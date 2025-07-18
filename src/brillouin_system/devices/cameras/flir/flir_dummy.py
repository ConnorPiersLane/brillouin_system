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

        print(f"[DummyFLIR] Initialized with ROI {width} x {height}")

    def configure_camera(self):
        pass

    def get_camera_info(self):
        return {
            "model": "DummyFLIR",
            "serial": "00000000",
            "sensor_size": self.get_sensor_size(),
            "roi": self.get_roi_native(),
            "gain": self.get_gain(),
            "exposure": self.get_exposure_time(),
            "pixel_format": self.get_pixel_format()
        }

    def set_resolution(self, width, height):
        self._roi = (0, 0, min(width, self._max_width), min(height, self._max_height))

    def get_resolution(self):
        return self._roi[2], self._roi[3]

    def get_sensor_size(self):
        return self._max_width, self._max_height

    def set_max_roi(self):
        self._roi = (0, 0, self._max_width, self._max_height)

    def set_roi_native(self, offset_x, offset_y, width, height):
        width = min(width, self._max_width)
        height = min(height, self._max_height)
        self._roi = (offset_x, offset_y, width, height)

    def get_roi_native(self):
        return self._roi

    def set_gain(self, value_dB):
        self._gain = value_dB

    def get_gain(self):
        return self._gain

    def set_gamma(self, value):
        pass

    def get_gamma(self):
        return None

    def set_exposure_time(self, value):
        self._exposure = value

    def get_exposure_time(self):
        return self._exposure

    def get_pixel_format(self):
        return self._pixel_format

    def set_pixel_format(self, format_str):
        self._pixel_format = format_str

    def start_single_frame_mode(self):
        self._is_single_frame_mode = True
        self._is_software_stream = False

    def acquire_image(self, timeout=1000):
        width, height = self.get_resolution()
        dtype = np.uint16 if self._pixel_format == 'Mono16' else np.uint8
        return np.zeros((height, width), dtype=dtype)

    def start_software_stream(self):
        self._is_software_stream = True
        self._is_single_frame_mode = False

    def software_snap_while_stream(self, timeout=1000):
        return self.acquire_image(timeout)

    def end_software_stream(self):
        self._is_software_stream = False

    def shutdown(self):
        print("[DummyFLIR] Shutdown called.")

    def get_available_pixel_formats(self):
        return ['Mono8', 'Mono16', 'Mono10Packed', 'Mono12Packed', 'Mono10p', 'Mono12p']

    def __del__(self):
        self.shutdown()
