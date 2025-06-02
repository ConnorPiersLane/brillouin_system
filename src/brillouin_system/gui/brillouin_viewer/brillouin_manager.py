import threading
import time
from typing import Callable

import numpy as np

from brillouin_system.config.config import calibration_config, CalibrationConfig, andor_frame_config
from brillouin_system.devices.cameras.andor.baseCamera import BaseCamera
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
from brillouin_system.devices.microwave_device import Microwave, MicrowaveDummy
from brillouin_system.devices.shutter_device import ShutterManager, ShutterManagerDummy
from brillouin_system.devices.zaber_linear_dummy import ZaberLinearDummy
from brillouin_system.my_dataclasses.background_image import ImageStatistics
from brillouin_system.utils.fit_util import get_px_and_sline_from_image
from brillouin_system.utils.calibration import CalibrationResults, CalibrationData, calibrate
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum, DisplayResults
from brillouin_system.my_dataclasses.measurements import MeasurementPoint, MeasurementSeries
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

        # Init Camera Settings:
        self.cam_settings_reference: CameraSettings = CameraSettings(
            name=self.camera.get_name(),
            exposure_time_s=0.2,
            emccd_gain=0,
            roi=self.camera.get_roi(),
            binning=self.camera.get_binning(),
            preamp_gain=self.camera.get_preamp_gain(),
            amp_mode=self.camera.get_amp_mode(),
        )
        self.cam_settings_sample: CameraSettings = self.get_camera_settings() # use init values here

        # State
        self.is_sample_illumination_continuous: bool = is_sample_illumination_continuous
        self.is_reference_mode: bool = False
        self.do_background_subtraction: bool = False
        self._is_do_bg_subtraction_selected_for_sample = False
        self.do_save_images: bool = False
        self.do_live_fitting = False


        # Calibration
        self.calibration_results: CalibrationResults | None = None

        # Dark Images
        self.dark_image_reference: ImageStatistics | None = None
        self.dark_image_sample: ImageStatistics | None = None
        # Background (BG) Image
        self.bg_image: ImageStatistics | None = None


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
        self.set_camera_settings(
            exposure_time=self.cam_settings_reference.exposure_time_s,
            emccd_gain=self.cam_settings_reference.emccd_gain,
        )
        print("[BrillouinManager] Switched to reference mode.")

    def change_to_sample_mode(self):
        self.is_reference_mode = False
        self.shutter_manager.change_to_objective()
        self.set_camera_settings(
            exposure_time=self.cam_settings_sample.exposure_time_s,
            emccd_gain=self.cam_settings_sample.emccd_gain,
        )
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
        return frame - self.bg_image.mean_image


    def take_n_images(self, n_images) -> np.ndarray:
        return np.stack([self._get_camera_snap() for _ in range(n_images)])

    def take_bg_image(self):
        """Capture and average multiple frames to use as background."""

        if self.is_sample_illumination_continuous:
            self.shutter_manager.sample.close()
        else:
            pass # shutter should already be closed
        time.sleep(0.05)  # Optional delay before acquisition

        andor_config = andor_frame_config.get()
        n_bg_images = andor_config.n_bg_images

        n_images = self.take_n_images(n_bg_images)

        if isinstance(self.camera, DummyCamera):
            n_images = n_images * 0.8

        self.bg_image = ImageStatistics(n_images)
        print("[BrillouinManager] Background Image acquired.")


        if self.is_sample_illumination_continuous:
            self.shutter_manager.sample.open()
        else:
            pass # do not open shutter, we are in snap mode

    def get_dark_image(self) -> ImageStatistics:
        andor_config = andor_frame_config.get()
        n_dark_images = andor_config.n_dark_images
        print(f"[BrillouinManager] Starting dark image acquisition, images to capture: {n_dark_images}")

        self.camera.close_shutter()
        time.sleep(0.1)

        n_images = self.take_n_images(n_dark_images)

        if isinstance(self.camera, DummyCamera):
            n_images = n_images * 0.01

        self.camera.open_shutter()
        time.sleep(0.05)

        print(f"[BrillouinManager] {n_dark_images} dark images acquired with: {self.get_camera_settings()}")

        return ImageStatistics(n_images)



    def is_background_image_available(self) -> bool:
        if self.bg_image is None:
            return False
        else:
            return True

    # ---------------- Get Frames  ----------------
    def _get_camera_snap(self) -> np.ndarray:
        """Pull a raw frame from the camera."""
        frame = self.camera.snap().astype(np.float64)
        return np.fliplr(frame)


    def get_andor_frame(self) -> np.ndarray:
        # Get the frame:
        if self.is_reference_mode or self.is_sample_illumination_continuous:
            frame = self._get_camera_snap()
        else:
            frame = self._open_sample_shutter_get_frame_close_shutter(timeout=1)

        return frame

    def get_fitted_spectrum(self, frame) -> FittedSpectrum:
        """
        Fits a Brillouin spectrum depending on reference mode and background subtraction.
        If live fitting is disabled, returns an unsuccessful fit but includes a raw spectrum line.

        Args:
            frame (np.ndarray): The input camera frame.

        Returns:
            FittedSpectrum: Dataclass containing fit results and metadata.
        """
        px, sline = get_px_and_sline_from_image(frame)

        if not self.do_live_fitting:
            #ToDo when do bg subtraction, frames should still be subtracted
            return FittedSpectrum(
                is_success=False,
                x_pixels=np.arange(sline.shape[0]),
                sline=sline,
            )

        try:
            # Fit the spectrum using the appropriate model
            if self.is_reference_mode:
                fitted_spectrum = brillouin_spectrum_fitting.get_fitted_spectrum_lorentzian(
                    px=px, sline=sline, is_reference_mode=True
                )
            elif self.do_background_subtraction:
                frame_with_sub_bg = self.subtract_background(frame)
                px, sline_bg_sub = get_px_and_sline_from_image(frame_with_sub_bg)
                fitted_spectrum = brillouin_spectrum_fitting.get_fitted_spectrum_lorentzian(
                    px=px, sline=sline_bg_sub, is_reference_mode=False
                )
            else:
                fitted_spectrum = brillouin_spectrum_fitting.get_fitted_spectrum_lorentzian(
                    px=px, sline=sline, is_reference_mode=False
                )
            return fitted_spectrum
        except Exception as e:
            print(f"[BrillouinManager] Fitting error: {e}")
            return FittedSpectrum(
                is_success=False,
                x_pixels=np.arange(sline.shape[0]),
                sline=sline,
            )

    def perform_calibration(self, config: CalibrationConfig, on_step: Callable[[DisplayResults], None]) -> bool:
        try:
            data = []
            for freq in config.calibration_freqs:
                freq_data = []
                self.microwave.set_frequency(freq)
                for _ in range(config.n_per_freq):
                    frame = self.get_andor_frame()
                    fs = self.get_fitted_spectrum(frame)
                    display = self.get_display_results(frame, fs)
                    on_step(display)
                    freq_data.append(fs)
                data.append(freq_data)

            cali_data = CalibrationData(freqs=config.calibration_freqs, n_per_freq=config.n_per_freq, fitted_spectras=data)
            self.calibration_results = calibrate(cali_data)
            return True
        except Exception as e:
            print(f"[Manager] Calibration failed: {e}")
            return False


    def take_measurements(self, n: int, which_axis: str, step: float, on_step: Callable[[DisplayResults], None]) -> MeasurementSeries:
        """
        Takes a series of measurements and calls on_step after each one to update the GUI.
        """
        measurements = []

        for i in range(n):
            try:
                self.log_message(f"Taking Measurement {i+1}")

                frame = self.get_andor_frame()
                fitting = self.get_fitted_spectrum(frame)
                display_results = self.get_display_results(frame, fitting)

                # Update GUI via provided callback
                on_step(display_results)

                result = self.get_measurement_data(frame, fitting)
                measurements.append(result)

                if which_axis and not self.is_reference_mode:
                    try:
                        self.zaber.move_rel(which_axis, step)
                        pos = self.zaber.get_position(which_axis)
                        self.log_message(f"Zaber {which_axis}-Axis moved by {step} µm to {pos:.2f} µm")
                    except Exception as e:
                        self.log_message(f"Zaber move failed: {e}")

            except Exception as e:
                self.log_message(f"[Measurement] Error at index {i}: {e}")

        return MeasurementSeries(measurements=measurements, calibration=self.calibration_results)

    def log_message(self, msg: str):
        """Optional helper to log messages — or use the signaller’s emit if needed."""
        print(f"[BrillouinManager] {msg}")

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


    def get_display_results(self, frame: np.ndarray, fitting: FittedSpectrum) -> DisplayResults:
        if self.is_reference_mode:
            freq_shift_ghz = self.microwave.get_frequency()
        elif fitting.is_success:
            freq_shift_ghz = self.compute_freq_shift(fitting)
        else:
            freq_shift_ghz = None

        if fitting.is_success:
            return DisplayResults(
                is_fitting_available=True,
                frame=frame,
                x_pixels=fitting.x_pixels,
                sline=fitting.sline,
                x_fit_refined=fitting.x_fit_refined,
                y_fit_refined=fitting.y_fit_refined,
                inter_peak_distance=fitting.inter_peak_distance,
                freq_shift_ghz=freq_shift_ghz,
            )
        else:
            return DisplayResults(
                is_fitting_available=False,
                frame=frame,
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

    def set_camera_settings(self,
                            exposure_time: float,
                            emccd_gain: int,
                            ):
        self.camera.set_exposure_time(exposure_time)
        self.camera.set_emccd_gain(emccd_gain)
        #ToDo: change ROI, binning from config

        andor_config = andor_frame_config.get()
        if andor_config.do_subtract_dark_image:
            dark_image = self.get_dark_image()
        else:
            dark_image = None

        self.camera.set_roi(x_start=andor_config.x_start,
                            x_end=andor_config.x_end,
                            y_start=andor_config.y_start,
                            y_end=andor_config.y_end,)
        self.camera.set_binning(hbin=andor_config.hbin,
                                vbin=andor_config.vbin)

        if self.is_reference_mode:
            self.dark_image_reference = dark_image
            self.cam_settings_reference: CameraSettings = self.get_camera_settings()
        else:
            self.dark_image_sample = dark_image
            self.cam_settings_sample: CameraSettings = self.get_camera_settings()





    def get_camera_settings(self) -> CameraSettings:
        return CameraSettings(name=self.camera.get_name(),
                              exposure_time_s=self.camera.get_exposure_time(),
                              emccd_gain=self.camera.get_emccd_gain(),
                              roi=self.camera.get_roi(),
                              binning=self.camera.get_binning(),
                              preamp_gain=self.camera.get_preamp_gain(),
                              amp_mode=self.camera.get_amp_mode())



    def get_measurement_data(self,
                             frame: np.ndarray,
                             fitting_results: FittedSpectrum) -> MeasurementPoint:

        if self.do_save_images:
            save_frame = frame
            bg_frame = self.bg_image
            dark_frame = self.dark_image_sample
        else:
            save_frame = None
            bg_frame = None
            dark_frame = None

        if self.is_reference_mode:
            zaber_position = None
        else:
            zaber_position = self.zaber.get_zaber_position_class()

        return MeasurementPoint(
            is_reference_mode=self.is_reference_mode,
            frame=save_frame,
            bg_frame=bg_frame,
            darknoise_frame=dark_frame,
            fitting_results = fitting_results,
            zaber_position=zaber_position,
            camera_settings=self.get_camera_settings(),
            mako_image=None,
            )


