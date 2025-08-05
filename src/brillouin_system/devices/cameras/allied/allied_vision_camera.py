from contextlib import ExitStack

from vimba import Vimba, VimbaFeatureError, VimbaCameraError
from brillouin_system.devices.cameras.allied.base_mako import BaseAlliedVisionCamera

class AlliedVisionCamera(BaseAlliedVisionCamera):
    def __init__(self, id="DEV_000F315BC084"):
        print("[AVCamera] Connecting to Allied Vision Camera...")
        self.stack = ExitStack()
        self.vimba = self.stack.enter_context(Vimba.get_instance())
        self.camera = None
        self.streaming = False
        self._last_callback = None

        cameras = self.vimba.get_all_cameras()
        if not cameras:
            raise RuntimeError("[AVCamera] No Allied Vision cameras found.")

        try:
            self.camera = self.stack.enter_context(self.vimba.get_camera_by_id(id))
        except VimbaCameraError:
            print(f"[AVCamera] Camera with ID '{id}' not found.")
            print("[AVCamera] Available cameras:")
            for cam in cameras:
                print(f"  - {cam.get_id()}")
            raise RuntimeError("[AVCamera] Cannot continue without valid camera.")

        print(f"[AVCamera] ...Found camera: {self.camera.get_id()}")
        self.set_freerun_mode()

    def set_freerun_mode(self):
        try:
            self.camera.TriggerSelector.set("FrameStart")
            self.camera.TriggerMode.set("Off")
            self.camera.AcquisitionMode.set("Continuous")
            print("[AVCamera] Camera set to Freerun mode.")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set Freerun mode: {e}")

    def set_software_trigger(self):
        try:
            self.camera.get_feature_by_name('TriggerSelector').set('FrameStart')
            self.camera.get_feature_by_name('TriggerMode').set('On')
            self.camera.get_feature_by_name('TriggerSource').set('Software')
            self.camera.get_feature_by_name('AcquisitionMode').set('Continuous')
            print("[AVCamera] Camera set to Software Trigger.")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set Software Trigger: {e}")

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

    def set_max_roi(self):
        """Set the camera to use the maximum allowable ROI."""
        max_roi = self.get_max_roi()
        if max_roi:
            self.set_roi(
                OffsetX=max_roi["OffsetX"],
                OffsetY=max_roi["OffsetY"],
                Width=max_roi["Width"],
                Height=max_roi["Height"]
            )
            print("[AVCamera] Set to max ROI.")
        else:
            print("[AVCamera] Could not retrieve max ROI to set.")

    def snap(self):
        """Capture a single frame. Temporarily stops streaming if needed."""
        was_streaming = self.streaming
        if was_streaming:
            self.stop_stream()

        self.set_acquisition_mode("SingleFrame")

        frame = self.camera.get_frame()

        if was_streaming:
            self.set_acquisition_mode("Continuous")
            self.start_stream(self._last_callback)

        return frame

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

        # Set to continuous mode
        self.set_acquisition_mode(mode="Continuous")

        def stream_handler(cam, frame):
            frame_callback(frame)
            cam.queue_frame(frame)

        # Allocate and queue initial frames

        self.camera.start_streaming(stream_handler, buffer_count=buffer_count)
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
        self.stack.close()
        print("[AVCamera] Camera and Vimba shut down.")

