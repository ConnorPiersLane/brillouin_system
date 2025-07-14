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



    def configure_camera(self):
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureTime.SetValue(20000.0)  # in microseconds
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(0.0)
        self.cam.GammaEnable.SetValue(False)
        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_SingleFrame)

    def get_camera_info(self):
        nodemap = self.cam.GetTLDeviceNodeMap()
        model = PySpin.CStringPtr(nodemap.GetNode("DeviceModelName")).GetValue()
        serial = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber")).GetValue()
        return {
            "model": model,
            "serial": serial,
            "sensor_size": self.get_sensor_size(),
            "roi": self.get_roi_native(),
            "gain": self.get_gain(),
            "exposure": self.get_exposure_time(),
            "pixel_format": self.get_pixel_format()
        }

    def set_resolution(self, width, height):
        sensor_width = self.cam.SensorWidth.GetValue()
        sensor_height = self.cam.SensorHeight.GetValue()

        offset_x = (sensor_width - width) // 2
        offset_y = (sensor_height - height) // 2

        self.cam.Width.SetValue(width)
        self.cam.Height.SetValue(height)
        self.cam.OffsetX.SetValue(offset_x)
        self.cam.OffsetY.SetValue(offset_y)

    def get_resolution(self):
        """Return current ROI size (width, height)."""
        return self.cam.Width.GetValue(), self.cam.Height.GetValue()

    def get_sensor_size(self):
        """Return sensor's full resolution (width, height)."""
        return self.cam.SensorWidth.GetValue(), self.cam.SensorHeight.GetValue()

    def set_max_roi(self):
        """Set ROI to maximum full sensor size, centered."""
        max_width = self.cam.SensorWidth.GetValue()
        max_height = self.cam.SensorHeight.GetValue()
        self.set_roi(max_width, max_height)

    def set_roi_native(self, offset_x, offset_y, width, height):
        """
        Set ROI using native-style arguments: offset and width/height.
        All values are aligned to the camera's required increment steps.
        """
        sensor_width = self.cam.SensorWidth.GetValue()
        sensor_height = self.cam.SensorHeight.GetValue()

        # Get increment steps
        offset_x_inc = self.cam.OffsetX.GetInc()
        offset_y_inc = self.cam.OffsetY.GetInc()
        width_inc = self.cam.Width.GetInc()
        height_inc = self.cam.Height.GetInc()

        # Align values
        aligned_offset_x = (offset_x // offset_x_inc) * offset_x_inc
        aligned_offset_y = (offset_y // offset_y_inc) * offset_y_inc
        aligned_width = (width // width_inc) * width_inc
        aligned_height = (height // height_inc) * height_inc

        # Validate range
        if aligned_offset_x + aligned_width > sensor_width:
            raise ValueError(f"ROI width exceeds sensor bounds ({sensor_width}px).")

        if aligned_offset_y + aligned_height > sensor_height:
            raise ValueError(f"ROI height exceeds sensor bounds ({sensor_height}px).")

        if aligned_width < width_inc or aligned_height < height_inc:
            raise ValueError(f"ROI must be at least one increment step: "
                             f"{width_inc} x {height_inc} pixels.")

        # Apply to camera
        self.cam.OffsetX.SetValue(aligned_offset_x)
        self.cam.OffsetY.SetValue(aligned_offset_y)
        self.cam.Width.SetValue(aligned_width)
        self.cam.Height.SetValue(aligned_height)

        def get_roi_native(self):
            """Return current ROI as (offset_x, offset_y, width, height)."""
            offset_x = self.cam.OffsetX.GetValue()
            offset_y = self.cam.OffsetY.GetValue()
            width = self.cam.Width.GetValue()
            height = self.cam.Height.GetValue()
            return offset_x, offset_y, width, height

    def set_gain(self, value_dB):
        """Set manual gain in dB (auto gain must be off)."""
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        min_gain = self.cam.Gain.GetMin()
        max_gain = self.cam.Gain.GetMax()

        if not (min_gain <= value_dB <= max_gain):
            raise ValueError(f"Gain must be between {min_gain} and {max_gain}")

        self.cam.Gain.SetValue(value_dB)

    def get_gain(self):
        """Return current manual gain value in dB."""
        return self.cam.Gain.GetValue()

    def set_gamma(self, value):
        """Enable and set gamma correction (typically between 0.1 and 4.0)."""
        if not PySpin.IsWritable(self.cam.GammaEnable):
            raise RuntimeError("Gamma control not supported or not writable.")

        min_gamma = self.cam.Gamma.GetMin()
        max_gamma = self.cam.Gamma.GetMax()

        if not (min_gamma <= value <= max_gamma):
            raise ValueError(f"Gamma value must be between {min_gamma} and {max_gamma}")

        self.cam.GammaEnable.SetValue(True)
        self.cam.Gamma.SetValue(value)

    def get_gamma(self):
        """Return current gamma value (or None if disabled)."""
        if not self.cam.GammaEnable.GetValue():
            return None
        return self.cam.Gamma.GetValue()

    def set_exposure_time(self, value):
        """Set manual exposure time in microseconds (auto exposure must be off)."""
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        min_exp = self.cam.ExposureTime.GetMin()
        max_exp = self.cam.ExposureTime.GetMax()

        if not (min_exp <= value <= max_exp):
            raise ValueError(f"Exposure time must be between {min_exp} and {max_exp}")

        self.cam.ExposureTime.SetValue(value)

    def get_exposure_time(self):
        """Return current manual exposure time in microseconds."""
        return self.cam.ExposureTime.GetValue()




    def get_pixel_format(self):
        """Return current pixel format as a string."""
        return self.cam.PixelFormat.GetCurrentEntry().GetSymbolic()

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

    def start_software_stream(self):
        # Set trigger mode to software
        self.cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
        self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)

        # Set acquisition mode to continuous
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        # Start acquisition
        self.cam.BeginAcquisition()

    def software_snap_while_stream(self, timeout=1000):
        self.cam.TriggerSoftware.Execute()
        image_result = self.cam.GetNextImage(timeout)

        if image_result.IsIncomplete():
            status = image_result.GetImageStatus()
            image_result.Release()
            raise RuntimeError(f"Incomplete image: status {status}")

        img = image_result.GetNDArray()
        image_result.Release()
        return img

    def end_software_stream(self):
        if self.cam.IsStreaming():
            self.cam.EndAcquisition()

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
        except Exception as e:
            print(f"[FLIRCamera] Shutdown exception ignored: {e}")


cam = FLIRCamera()
try:
    img = cam.acquire_image()
    cv2.imshow("FLIR Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
finally:
    cam.shutdown()