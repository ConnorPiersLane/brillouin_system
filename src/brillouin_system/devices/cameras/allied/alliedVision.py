from brillouin_system.devices import BrillouinDevice
import numpy as np
from vimba import Vimba, VimbaFeatureError
from PyQt5.QtCore import pyqtSignal

class AlliedVision(BrillouinDevice.Device):
    def __init__(self, stop_event, app):
        super(AlliedVision, self).__init__(stop_event)
        self.deviceName = "Mako"
        self.camera = None

        # Initialize Vimba and manually enter its context.
        self.vimba = Vimba.get_instance()
        self.vimba.__enter__()

        cameras = self.vimba.get_all_cameras()
        if not cameras:
            raise RuntimeError("[MakoDevice] No Mako camera found!")
        self.camera = cameras[0]

        # Manually enter the camera context so its features remain accessible.
        self.camera.__enter__()
        print(f"[MakoDevice] CMOS camera found: {self.camera.get_id()}")

        self.set_up()
        self.camera.start_streaming(self.frame_handler)

        self.mako_lock = app.mako_lock
        self.runMode = 0  # 0 = free running, 1 = scan

    def set_up(self):
        # Set acquisition mode to SingleFrame.
        self.camera.get_feature_by_name("AcquisitionMode").set("SingleFrame")
        exposure_feature = None
        for name in ["ExposureTimeAbs", "ExposureTimeUs", "ExposureTimeRaw"]:
            try:
                self.camera.get_feature_by_name(name).set(20000)  # Default exposure: 20000 µs.
                print(f"[MakoDevice] Set exposure using feature: {name}")
                exposure_feature = name
                break
            except VimbaFeatureError:
                continue
        if exposure_feature is None:
            raise RuntimeError("[MakoDevice] No valid exposure time feature found!")
        self.imageHeight = 800
        self.imageWidth = 800
        self.bin_size = 1
        self.camera.get_feature_by_name("Height").set(self.imageHeight)
        self.camera.get_feature_by_name("Width").set(self.imageWidth)
        self.camera.get_feature_by_name("OffsetX").set(624)
        self.camera.get_feature_by_name("OffsetY").set(624)

    def frame_handler(self, cam, frame):
        """Handles asynchronous frames during streaming."""
        print(f"[MakoDevice] Frame received: {frame.get_id()}")

    def shutdown(self):
        print("[MakoDevice] Closing Device")
        try:
            self.camera.stop_streaming()
        except Exception as e:
            print("[MakoDevice] Error stopping streaming:", e)
        self.camera.__exit__(None, None, None)
        self.vimba.__exit__(None, None, None)

    def getData(self):
        """
        Acquire a single frame synchronously.
        Since synchronous acquisition (via get_frame()) cannot be done while streaming,
        we temporarily stop streaming, grab a frame, and then restart streaming.
        The returned image will be a 2D array (height, width) as in the reference code.
        """
        with self.mako_lock:
            try:
                self.camera.stop_streaming()
            except Exception as e:
                print("[MakoDevice] Error stopping streaming for synchronous acquisition:", e)
            try:
                frame = self.camera.get_frame()
                frame.convert_pixel_format(frame.get_pixel_format())
                image_arr = frame.as_numpy_ndarray()
                # If the image array has a singleton third dimension, squeeze it.
                if image_arr.ndim == 3 and image_arr.shape[-1] == 1:
                    image_arr = image_arr[..., 0]
            except Exception as e:
                print("[MakoDevice] Timed out while waiting for new frame:", e)
                image_arr = None
            try:
                self.camera.start_streaming(self.frame_handler)
            except Exception as e:
                print("[MakoDevice] Error restarting streaming:", e)
            return image_arr

    def setExpTime(self, expTime):
        """Set the exposure time (in ms; converted internally to µs)."""
        with self.mako_lock:
            try:
                self.camera.get_feature_by_name("ExposureTimeAbs").set(expTime * 1e3)
            except Exception as e:
                print("Error setting exposure time:", e)

    def setFrameRate(self, frameRate):
        """Set the camera frame rate using 'AcquisitionFrameRateAbs' feature."""
        with self.mako_lock:
            try:
                self.camera.get_feature_by_name("AcquisitionFrameRateAbs").set(frameRate)
            except Exception as e:
                print("Error setting frame rate:", e)

class MakoFreerun(BrillouinDevice.DeviceProcess):
    updateCMOSImageSig = pyqtSignal('PyQt_PyObject')

    def __init__(self, device, stopProcessingEvent, finishedTrigger=None):
        super(MakoFreerun, self).__init__(device, stopProcessingEvent, finishedTrigger)

    def doComputation(self, data):
        if data is None:
            print("[MakoDevice] Warning: Received None frame, skipping computation.")
            return None  # Skip processing if the frame is invalid
        try:
            image = np.flip(data.transpose((1, 0)), 1)
            self.updateCMOSImageSig.emit(image)
            return image
        except Exception as e:
            print(f"[MakoDevice] Error in doComputation: {e}")
            return None
