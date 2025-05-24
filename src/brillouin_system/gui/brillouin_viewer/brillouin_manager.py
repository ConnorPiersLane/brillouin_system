import threading
import time

import numpy as np

from brillouin_system.config.config import calibration_config
from brillouin_system.devices.cameras.andor.baseCamera import BaseCamera
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
from brillouin_system.devices.microwave_device import Microwave, MicrowaveDummy
from brillouin_system.devices.shutter_device import ShutterManager, ShutterManagerDummy
from brillouin_system.devices.zaber_linear_dummy import ZaberLinearDummy
from brillouin_system.my_dataclasses.calibration import CalibrationResults, CalibrationData, calibrate
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum, DisplayResults
from brillouin_system.my_dataclasses.measurement_data import MeasurementData
from brillouin_system.my_dataclasses.camera_settings import CameraSettings
from brillouin_system.utils import brillouin_spectrum_fitting
from brillouin_system.devices.zaber_linear import ZaberLinearController



def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 using contrast stretching."""
    img = np.copy(image)
    img = np.clip(img, 0, np.percentile(img, 99))  # contrast stretch
    img = 255 * (img / img.max()) if img.max() > 0 else img
    return img.astype(np.uint8)


class BrillouinManager:

    def __init__(self,
                 camera: BaseCamera | DummyCamera,
                 shutter_manager: ShutterManager | ShutterManagerDummy,
                 microwave: Microwave | MicrowaveDummy,
                 zaber: ZaberLinearController | ZaberLinearDummy,
                 is_sample_illumination_continuous: bool = False,
                 ):
        # Devices
        self.camera: BaseCamera | DummyCamera = camera
        self.shutter_manager: ShutterManager | ShutterManagerDummy = shutter_manager
        self.microwave: Microwave | MicrowaveDummy = microwave
        self.zaber = zaber


        # State
        self.is_sample_illumination_continuous: bool = is_sample_illumination_continuous
        self.is_reference_mode: bool = False
        self.do_background_subtraction: bool = False
        self.do_save_images: bool = False


        # Calibration
        self.calibration_results: CalibrationResults = None

        self.bg_image = None

        self.init_shutters()

    def init_shutters(self):
        if self.is_reference_mode:
            self.shutter_manager.change_to_reference()
        else:
            self.shutter_manager.change_to_objective()
        if self.is_sample_illumination_continuous:
            self.shutter_manager.sample.open()
        else:
            self.shutter_manager.sample.close()



    # ---------------- Change Modes ----------------

    def change_illumination_mode_to_continuous(self):
        self.is_sample_illumination_continuous = True
        self.shutter_manager.sample.open()
        print("[BrillouinManager] Switched to continuous illumination mode.")

    def change_illumination_mode_to_pulsed(self):
        self.is_sample_illumination_continuous = False
        self.shutter_manager.sample.close()
        print("[BrillouinManager] Switched to pulsed illumination mode.")

    def change_to_reference_mode(self):
        self.is_reference_mode = True
        self.shutter_manager.change_to_reference()
        print("[BrillouinManager] Switched to reference mode.")

    def change_to_sample_mode(self):
        self.is_reference_mode = False
        self.shutter_manager.change_to_objective()
        print("[BrillouinManager] Switched to sample mode.")

    # ----------------- Background Subtraction ----------------- #

    def start_background_subtraction(self):
        if self.is_background_image_available():
            self.do_background_subtraction = True
        else:
            self.do_background_subtraction = False
            print("[BrillouinManager] No Background Image available")

    def stop_background_subtraction(self):
        self.do_background_subtraction = False

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        if not self.is_background_image_available():
            print("[AcquisitionManager] No background image available")
            return frame
        return frame - self.bg_image


    def acquire_background_image(self, number_of_images_to_be_taken: int):
        """Capture and average multiple frames to use as background."""

        if self.is_reference_mode:
            self.shutter_manager.reference.close()
        else:
            if self.is_sample_illumination_continuous:
                self.shutter_manager.sample.close()
            else:
                pass # shutter should already be closed
        time.sleep(0.05)  # Optional delay before acquisition

        if isinstance(self.camera, DummyCamera):
            time.sleep(5)
            self.bg_image = self._get_camera_snap() * 0.99  # simulate something
        else:
            self.bg_image = np.mean(
                np.stack([self._get_camera_snap() for _ in range(number_of_images_to_be_taken)]),
                axis=0
            )
        print("[BrillouinManager] Background Image acquired.")

        if self.is_reference_mode:
            self.shutter_manager.reference.open()
        else:
            if self.is_sample_illumination_continuous:
                self.shutter_manager.sample.open()
            else:
                pass # do not open shutter, we are in snap mode

    def is_background_image_available(self) -> bool:
        return self.bg_image is not None

    # ---------------- Get Frames  ----------------
    def _get_camera_snap(self) -> np.ndarray:
        """Pull a raw frame from the camera."""
        return self.camera.snap().astype(np.float64)

    def _get_frame(self) -> np.ndarray:
        # Get the frame:
        if self.is_reference_mode or self.is_sample_illumination_continuous:
            frame = self._get_camera_snap()
        else:
            frame = self._open_sample_shutter_get_frame_close_shutter(timeout=1)

        if self.do_background_subtraction:
            frame = self.subtract_background(frame)

        return frame

    def update_calibration(self, calibration_data: CalibrationData):
        self.calibration_results = calibrate(calibration_data)
        print("[Manager] Calibration updated")



    def snap_and_get_fitting(self) -> FittedSpectrum:
        frame = self._get_frame()

        fitted_spectrum: FittedSpectrum = brillouin_spectrum_fitting.get_fitted_spectrum_from_image(
            frame=frame, is_reference_mode=self.is_reference_mode
        )

        return fitted_spectrum

    def compute_freq_shift(self, fitting: FittedSpectrum) -> float | None:
        if not fitting.is_success or self.calibration_results is None:
            return None

        config = calibration_config.get()
        calibration = self.calibration_results.get_calibration(config)

        if config.reference == "left":
            x_value = fitting.left_peak_center_px
        elif config.reference == "right":
            x_value = fitting.right_peak_center_px
        elif config.reference == "distance":
            x_value = fitting.inter_peak_distance
        else:
            return None

        if x_value is None or np.isnan(x_value):
            return None

        return float(calibration.get_freq(x_value))


    def get_display_results_from_fitting(self, fitting: FittedSpectrum) -> DisplayResults:
        if self.is_reference_mode:
            freq_shift_ghz = self.microwave.get_frequency()
        elif fitting.is_success:
            freq_shift_ghz = self.compute_freq_shift(fitting)
        else:
            freq_shift_ghz = None

        if fitting.is_success:
            return DisplayResults(
                is_success=True,
                frame=fitting.frame,
                x_pixels=fitting.x_pixels,
                sline=fitting.sline,
                x_fit_refined=fitting.x_fit_refined,
                y_fit_refined=fitting.y_fit_refined,
                inter_peak_distance=fitting.inter_peak_distance,
                freq_shift_ghz=freq_shift_ghz,
            )
        else:
            return DisplayResults(
                is_success=False,
                frame=fitting.frame,
                x_pixels=fitting.x_pixels,
                sline=fitting.sline,
            )


    def _open_sample_shutter_get_frame_close_shutter(self, timeout: float) -> np.ndarray:
        """Acquire a frame with temporary shutter open for pulsed mode."""
        frame = None
        timer = threading.Timer(timeout, self.shutter_manager.sample.close)

        try:
            self.shutter_manager.sample.open()
            timer.start()
            frame = self._get_camera_snap()
        finally:
            timer.cancel()
            self.shutter_manager.sample.close()

        return frame

    def get_camera_settings(self) -> CameraSettings:
        return CameraSettings(name=self.camera.get_name(),
                                           exposure_time_s=self.camera.get_exposure_time(),
                                           gain=self.camera.get_gain(),
                                           roi=self.camera.get_roi(),
                                           binning=self.camera.get_binning())



    def get_measurement_data(self,
                             fitting_results: FittedSpectrum) -> MeasurementData:

        if self.do_save_images:
            pass
        else:
            fitting_results.frame = None

        if self.is_reference_mode:
            zaber_position = None
        else:
            zaber_position = self.zaber.get_zaber_position_class()

        return MeasurementData(
            is_reference_mode=self.is_reference_mode,
            fitting_results = fitting_results,
            calibration = self.calibration_results,
            zaber_position=zaber_position,
            camera_settings=self.get_camera_settings(),
            mako_image=None,
            )


