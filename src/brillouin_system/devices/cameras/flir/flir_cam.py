# flir_wrapper.py

import PySpin
import cv2
import numpy as np


class FLIRCamera:
    def __init__(self, index=0, width=2000, height=2000):
        '''

        Args:
            index:
            width: max 3216
            height: max 2208
        '''
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() <= index:
            raise RuntimeError("No FLIR camera found at index {}".format(index))

        self.cam = self.cam_list.GetByIndex(index)
        self.cam.Init()

        self.set_resolution(width, height)
        self.configure_camera()
        sensor_width = self.cam.SensorWidth.GetValue()
        sensor_height = self.cam.SensorHeight.GetValue()

        print(f"Full sensor resolution: {sensor_width} x {sensor_height}")

    def set_resolution(self, width, height):
        sensor_width = self.cam.SensorWidth.GetValue()
        sensor_height = self.cam.SensorHeight.GetValue()

        offset_x = (sensor_width - width) // 2
        offset_y = (sensor_height - height) // 2

        self.cam.Width.SetValue(width)
        self.cam.Height.SetValue(height)
        self.cam.OffsetX.SetValue(offset_x)
        self.cam.OffsetY.SetValue(offset_y)

    def configure_camera(self):
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureTime.SetValue(20000.0)  # in microseconds
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(0.0)
        self.cam.GammaEnable.SetValue(False)
        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_SingleFrame)

    def acquire_image(self, timeout=1000):
        self.cam.BeginAcquisition()
        image_result = self.cam.GetNextImage(timeout)

        if image_result.IsIncomplete():
            image_result.Release()
            raise RuntimeError("Image incomplete with status {}".format(image_result.GetImageStatus()))

        img_array = image_result.GetNDArray()
        image_result.Release()
        self.cam.EndAcquisition()
        return img_array

    def shutdown(self):
        if self.cam.IsStreaming():
            self.cam.EndAcquisition()

        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

cam = FLIRCamera()
try:
    img = cam.acquire_image()
    cv2.imshow("FLIR Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
finally:
    cam.shutdown()