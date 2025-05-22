from vimba import Vimba, VimbaFeatureError
import numpy as np
from .base_mako import BaseMakoCamera

class AlliedVisionCamera(BaseMakoCamera):
    def __init__(self):
        print("[AVCamera] Connecting to Allied Vision Camera...")
        self.vimba = Vimba.get_instance()
        self.vimba.__enter__()
        self.camera = None
        self.streaming = False

        cameras = self.vimba.get_all_cameras()
        if not cameras:
            raise RuntimeError("No Allied Vision camera found.")
        self.camera = cameras[0]
        self.camera.__enter__()
        print(f"[AVCamera] ...Found camera: {self.camera.get_id()}")

    def set_exposure(self, exposure_us):
        feat = self.camera.get_feature_by_name("ExposureTimeAbs")
        feat.set(exposure_us)
        print(f"[AVCamera] Exposure time set to {exposure_us} us")

    def get_exposure(self):
        return self.camera.get_feature_by_name("ExposureTimeAbs").get()

    def set_auto_exposure(self, mode='Off'):
        """
        Set the auto exposure mode.
        mode: 'Off', 'Once', or 'Continuous'
        """
        try:
            feat = self.camera.get_feature_by_name("ExposureAuto")
            feat.set(mode)
            print(f"[AVCamera] Auto exposure mode set to {mode}")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set Auto Exposure mode: {e}")

    def get_auto_exposure(self):
        try:
            return str(self.camera.get_feature_by_name("ExposureAuto").get())
        except VimbaFeatureError:
            print("[AVCamera] Auto Exposure feature not available.")
            return None

    def set_gain(self, gain_db):
        try:
            self.camera.get_feature_by_name("Gain").set(gain_db)
            print(f"[AVCamera] Gain set to {gain_db} dB")
        except VimbaFeatureError:
            print("[AVCamera] Gain setting not supported or failed.")

    def get_gain(self):
        return self.camera.get_feature_by_name("Gain").get()

    def set_roi(self, OffsetX, OffsetY, Width, Height):
        try:
            self.camera.get_feature_by_name("OffsetX").set(OffsetX)
            self.camera.get_feature_by_name("OffsetY").set(OffsetY)
            self.camera.get_feature_by_name("Width").set(Width)
            self.camera.get_feature_by_name("Height").set(Height)
            print(f"[AVCamera] ROI set to x:{OffsetX} y:{OffsetY} width:{Width} height:{Height}")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set ROI: {e}")

    def get_roi(self):
        return {
            "OffsetX": self.camera.get_feature_by_name("OffsetX").get(),
            "OffsetY": self.camera.get_feature_by_name("OffsetY").get(),
            "Width": self.camera.get_feature_by_name("Width").get(),
            "Height": self.camera.get_feature_by_name("Height").get(),
        }

    def get_max_roi(self):
        try:
            width_feat = self.camera.get_feature_by_name("Width")
            height_feat = self.camera.get_feature_by_name("Height")
            offsetx_feat = self.camera.get_feature_by_name("OffsetX")
            offsety_feat = self.camera.get_feature_by_name("OffsetY")

            max_width = width_feat.get_range()[1]  # (min, max)
            max_height = height_feat.get_range()[1]
            min_offset_x = offsetx_feat.get_range()[0]
            min_offset_y = offsety_feat.get_range()[0]

            return {
                "OffsetX": min_offset_x,
                "OffsetY": min_offset_y,
                "Width": max_width,
                "Height": max_height
            }
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to get max ROI: {e}")
            return None

    def snap(self):
        """Capture a single frame. Temporarily stops streaming if needed."""
        was_streaming = self.streaming
        if was_streaming:
            self.stop_stream()

        self.set_acquisition_mode("SingleFrame")

        frame = self.camera.get_frame()
        frame.convert_pixel_format(frame.get_pixel_format())
        image = frame.as_numpy_ndarray()
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        if was_streaming:
            self.set_acquisition_mode("Continuous")
            self.start_stream(self._last_callback)

        return image

    def set_acquisition_mode(self, mode="Continuous"):
        """
        "SingleFrame" or "Continuous"
        """
        try:
            self.camera.get_feature_by_name("AcquisitionMode").set(mode)
            print(f"[AVCamera] Acquisition mode set to {mode}")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set AcquisitionMode to {mode}: {e}")

    def start_stream(self, frame_callback, buffer_count=5):
        if self.streaming:
            print("[AVCamera] Already streaming.")
            return

        self._last_callback = frame_callback

        # ✅ Set to continuous mode
        self.set_acquisition_mode(mode="Continuous")

        def stream_handler(cam, frame):
            frame_callback(frame.as_numpy_ndarray())
            cam.queue_frame(frame)

        # Allocate and queue initial frames
        self.frames = [self.camera.get_frame() for _ in range(buffer_count)]
        for frame in self.frames:
            self.camera.queue_frame(frame)

        self.camera.start_streaming(stream_handler)
        self.streaming = True
        print("[AVCamera] Started streaming.")

    def stop_stream(self):
        if not self.streaming:
            print("[AVCamera] Not currently streaming.")
            return
        self.camera.stop_streaming()
        self.streaming = False
        print("[AVCamera] Stopped streaming.")


    def close(self):
        if self.streaming:
            self.stop_stream()
        self.camera.__exit__(None, None, None)
        self.vimba.__exit__(None, None, None)
        print("[AVCamera] Camera and Vimba shut down.")
