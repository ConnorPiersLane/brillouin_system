import threading

import numpy as np
import time

from brillouin_system.devices.cameras.allied.allied_config.allied_config import AlliedConfig
from brillouin_system.devices.cameras.allied.single.base_allied_vision_camera import BaseAlliedVisionCamera



class DummyAlliedVisionCamera(BaseAlliedVisionCamera):
    def __init__(self):
        self.streaming = False
        self._last_callback = None
        self._exposure = 10000  # in µs
        self._gain = 10.0       # in dB
        self._roi = {
            "OffsetX": 0,
            "OffsetY": 0,
            "Width": 640,
            "Height": 480
        }
        print("[DummyMakoCamera] Initialized dummy camera.")

    def set_exposure(self, exposure_us):
        self._exposure = exposure_us
        print(f"[DummyMakoCamera] Exposure set to {exposure_us} µs")

    def get_exposure(self):
        return self._exposure

    def set_gain(self, gain_db):
        self._gain = gain_db
        print(f"[DummyMakoCamera] Gain set to {gain_db} dB")

    def get_gain(self):
        return self._gain

    def set_roi(self, offset_x, offset_y, width, height):
        self._roi = {
            "OffsetX": offset_x,
            "OffsetY": offset_y,
            "Width": width,
            "Height": height
        }
        print(f"[DummyMakoCamera] ROI set to x:{offset_x} y:{offset_y} width:{width} height:{height}")

    def get_roi(self):
        return self._roi.copy()

    def set_acquisition_mode(self, mode="Continuous"):
        print(f"[DummyMakoCamera] Acquisition mode set to {mode}")

    def start_stream(self, frame_callback, buffer_count=5):
        if self.streaming:
            print("[DummyMakoCamera] Already streaming.")
            return

        self.streaming = True
        self._last_callback = frame_callback
        print("[DummyMakoCamera] Started streaming.")

        def stream_loop():
            frame_id = 0
            while self.streaming:
                dummy_frame = self._generate_dummy_frame(frame_id)
                frame_callback(dummy_frame)
                frame_id += 1
                time.sleep(0.1)  # ~10 FPS

        self._thread = threading.Thread(target=stream_loop, daemon=True)
        self._thread.start()

    def stop_stream(self):
        if self.streaming:
            print("[DummyMakoCamera] Stopped streaming.")
            self.streaming = False

    def set_config(self, cfg: AlliedConfig):
        self.set_roi(cfg.offset_x, cfg.offset_y, cfg.width, cfg.height)
        self.set_exposure(cfg.exposure)
        self.set_gain(cfg.gain)
        print(f"[DummyMakoCamera] Config applied for {cfg.id}")

    def snap(self):
        was_streaming = self.streaming
        if was_streaming:
            self.stop_stream()

        self.set_acquisition_mode("SingleFrame")

        image = self._generate_dummy_frame()

        if was_streaming and self._last_callback:
            self.set_acquisition_mode("Continuous")
            self.start_stream(self._last_callback)

        return image

    def _generate_dummy_frame(self, frame_id=0):
        shape = (self._roi["Height"], self._roi["Width"])
        data = np.random.randint(0, 255, shape, dtype=np.uint8)
        return data

    def close(self):
        if self.streaming:
            self.stop_stream()
        print("[DummyMakoCamera] Camera shut down.")
