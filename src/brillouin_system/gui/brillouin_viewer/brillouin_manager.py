import threading
import time
from typing import Callable

import numpy as np

from brillouin_system.config.config import CalibrationConfig, andor_frame_config, calibration_config
from brillouin_system.devices.cameras.andor.baseCamera import BaseCamera
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
from brillouin_system.devices.microwave_device import Microwave, MicrowaveDummy
from brillouin_system.devices.shutter_device import ShutterManager, ShutterManagerDummy
from brillouin_system.devices.zaber_linear_dummy import ZaberLinearDummy
from brillouin_system.fitting.compute_sample_freqs import compute_freq_shift
from brillouin_system.fitting.fitting_manager import get_empty_fitting, fit_reference_spectrum, fit_sample_spectrum
from brillouin_system.my_dataclasses.background_image import ImageStatistics, generate_image_statistics_dataclass
from brillouin_system.my_dataclasses.state_mode import StateMode
from brillouin_system.my_dataclasses.zaber_position import generate_zaber_positions
from brillouin_system.fitting.fit_util import get_sline_from_image
from brillouin_system.my_dataclasses.calibration import CalibrationData, \
    CalibrationMeasurementPoint, MeasurementsPerFreq, CalibrationCalculator, get_calibration_calculator_from_data
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum, DisplayResults
from brillouin_system.my_dataclasses.measurements import MeasurementPoint, MeasurementSeries, MeasurementSettings
from brillouin_system.my_dataclasses.camera_settings import AndorCameraSettings
from brillouin_system.devices.zaber_linear import ZaberLinearController



def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 using contrast stretching."""
    img = np.copy(image)
    img = np.clip(img, 0, np.percentile(img, 99))  # contrast stretch
    img = 255 * (img / img.max()) if img.max() > 0 else img
    return img.astype(np.uint8)


class BrillouinManager:

    @staticmethod
    def log_message(msg: str):
        """Optional helper to log messages — or use the signaller’s emit if needed."""
        print(f"[BrillouinManager] {msg}")

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
        self._is_do_bg_subtraction_selected_for_sample = False
        self.do_live_fitting = False

        # Calibration
        self.calibration_data: CalibrationData | None = None
        self.calibration_calculator: CalibrationCalculator | None = None

        # Background (BG) Image and dark_image for the sample
        self.bg_image: ImageStatistics | None = None
        self.dark_image: ImageStatistics | None = None

        self.init_shutters()
        self.init_camera_settings()

        # Init state modes
        self.reference_state_mode: StateMode = self.init_state_mode(is_reference_mode=True)
        self.sample_state_mode: StateMode = self.init_state_mode(is_reference_mode=False)

    def init_shutters(self):
        if self.is_reference_mode:
            self.shutter_manager.change_to_reference()
        else:
            self.shutter_manager.change_to_objective()
        if self.is_sample_illumination_continuous:
            self.shutter_manager.sample.open()
        else:
            self.shutter_manager.sample.close()

    def init_camera_settings(self):
        andor_config = andor_frame_config.get()
        self.camera.set_roi(x_start=andor_config.x_start,
                            x_end=andor_config.x_end,
                            y_start=andor_config.y_start,
                            y_end=andor_config.y_end,)
        self.camera.set_binning(hbin=andor_config.hbin,
                                vbin=andor_config.vbin)


    def init_state_mode(self, is_reference_mode: bool) -> StateMode:
        return StateMode(
            is_reference_mode=is_reference_mode,
            is_do_bg_subtraction_active=False,
            bg_image=None,
            dark_image=None,
            camera_settings=self.get_andor_camera_settings()
        )

    # ---------------- Change Modes ----------------

    def change_illumination_mode_to_continuous(self):
        self.is_sample_illumination_continuous = True
        self.shutter_manager.sample.open()
        print("[BrillouinManager] Switched to continuous illumination mode.")

    def change_illumination_mode_to_pulsed(self):
        self.is_sample_illumination_continuous = False
        self.shutter_manager.sample.close()
        print("[BrillouinManager] Switched to pulsed illumination mode.")

    def change_state_modes(self, state_mode: StateMode):
        self.is_reference_mode = state_mode.is_reference_mode
        self.do_background_subtraction = state_mode.is_do_bg_subtraction_active
        self.bg_image = state_mode.bg_image
        self.set_camera_settings(
            exposure_time=state_mode.camera_settings.exposure_time_s,
            emccd_gain=state_mode.camera_settings.emccd_gain,
        )

    def get_current_state_mode(self) -> StateMode:
        return StateMode(
            is_reference_mode=self.is_reference_mode,
            is_do_bg_subtraction_active=self.do_background_subtraction,
            bg_image=self.bg_image,
            dark_image=self.dark_image,
            camera_settings=self.get_andor_camera_settings()
        )

    def change_to_reference_mode(self):
        # Store current state mode of the sample for future:
        self.sample_state_mode = self.get_current_state_mode()

        self.shutter_manager.change_to_reference()
        self.change_state_modes(state_mode=self.reference_state_mode)
        print("[BrillouinManager] Switched to reference mode.")


    def change_to_sample_mode(self):
        # Store current state mode of the sample for future:
        self.reference_state_mode = self.get_current_state_mode()
        self.shutter_manager.change_to_objective()
        self.change_state_modes(state_mode=self.sample_state_mode)
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
        result = frame - self.bg_image.mean_image
        result = np.clip(result, 0, None)  # enforce non-negativity
        return result


    def take_n_images(self, n_images) -> np.ndarray:
        return np.stack([self._get_camera_snap() for _ in range(n_images)])


    def take_bg_and_darknoise_images(self):

        self.dark_image: ImageStatistics = self.get_dark_image()
        self.bg_image: ImageStatistics = self.get_bg_image()


    def get_bg_image(self):
        """Capture and average multiple frames to use as background."""

        if self.is_sample_illumination_continuous:
            self.shutter_manager.sample.close()
        else:
            pass # shutter should already be closed
        time.sleep(0.05)  # Optional delay before acquisition

        andor_config = andor_frame_config.get()

        n_bg_images = andor_config.n_bg_images
        print(f"[BrillouinManager] Taking {n_bg_images} Background Images...")
        n_images = self.take_n_images(n_bg_images)

        if isinstance(self.camera, DummyCamera):
            n_images = n_images * 0.8


        print("[BrillouinManager] ...Background Images acquired.")


        if self.is_sample_illumination_continuous:
            self.shutter_manager.sample.open()
        else:
            pass # do not open shutter, we are in snap mode

        return generate_image_statistics_dataclass(n_images)


    def get_dark_image(self) -> ImageStatistics | None:
        andor_config = andor_frame_config.get()

        if not andor_config.take_dark_image:
            return None

        n_dark_images = andor_config.n_dark_images
        print(f"[BrillouinManager] Starting dark image acquisition, images to capture: {n_dark_images}")

        self.camera.close_shutter()
        time.sleep(0.1)

        n_images = self.take_n_images(n_dark_images)

        if isinstance(self.camera, DummyCamera):
            n_images = n_images * 0.01

        self.camera.open_shutter()
        time.sleep(0.05)

        print(f"[BrillouinManager] {n_dark_images} dark images acquired with: {self.get_andor_camera_settings()}")

        return generate_image_statistics_dataclass(n_images)



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

        if self.do_background_subtraction:
            frame_with_sub_bg = self.subtract_background(frame)
            sline = get_sline_from_image(frame_with_sub_bg)
        else:
            sline = get_sline_from_image(frame)

        if not self.do_live_fitting:
            return get_empty_fitting(sline)

        try:
            if self.is_reference_mode:
                return fit_reference_spectrum(sline=sline)
            else:
                return fit_sample_spectrum(sline=sline, calibration_calculator=self.calibration_calculator)
        except Exception as e:
            print(f"[BrillouinManager] Fitting error: {e}")
            return get_empty_fitting(sline)

    def update_calibration_calculator(self):
        if self.calibration_data is None:
            self.calibration_calculator = None
        else:
            self.calibration_calculator: CalibrationCalculator = get_calibration_calculator_from_data(self.calibration_data)



    def perform_calibration(self, config: CalibrationConfig, call_update_gui: Callable[[DisplayResults], None]) -> bool:
        """
        Perform a calibration measurement series and store the results.

        Args:
            config: CalibrationConfig with frequencies and number of points
            call_update_gui: Callback to update GUI after each measurement

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            measured_freqs = []
            for freq in config.calibration_freqs:
                self.microwave.set_frequency(freq)
                freq_points = []
                for _ in range(config.n_per_freq):
                    frame = self.get_andor_frame()
                    fs = self.get_fitted_spectrum(frame)
                    calibration_point = CalibrationMeasurementPoint(
                        frame=frame,
                        microwave_freq=self.microwave.get_frequency(),  # measured freq from device
                        fitting_results=fs,
                    )
                    call_update_gui(self.get_display_results(frame, fs))
                    freq_points.append(calibration_point)
                measured_freqs.append(MeasurementsPerFreq(set_freq_ghz=freq,
                                                          state_mode=self.get_current_state_mode(),
                                                          cali_meas_points=freq_points))

            self.calibration_data = CalibrationData(measured_freqs=measured_freqs)
            self.update_calibration_calculator()
            return True
        except Exception as e:
            print(f"[Manager] Calibration failed: {e}")
            return False


    def take_one_measurement(self, zaber_position) -> MeasurementPoint:
        self.zaber.set_zaber_position_by_class(zaber_position=zaber_position)

        frame = self.get_andor_frame()

        return MeasurementPoint(
            frame=frame,
            zaber_position=self.zaber.get_zaber_position_class(),
            mako_image=None,
        )

    def take_measurement_series(
            self,
            measurement_settings: MeasurementSettings,
            call_update_gui: Callable[[DisplayResults], None]
    ) -> MeasurementSeries:
        measurements = []

        # Generate ZaberPositions
        # Current position:
        which_axis = measurement_settings.move_axes  # ToDo: make this versatil for other axex
        # Todo: save images is not beeing used

        start = self.zaber.get_position(which_axis)  # or any default you want
        step = measurement_settings.move_x_rel_um
        n = measurement_settings.n_measurements

        fixed_positions = {}  # optionally set this if you have known values

        zaber_positions = generate_zaber_positions(
            axis='x',
            start=start,
            step=step,
            n=n,
            fixed_positions=fixed_positions
        )


        for i, zaber_pos in enumerate(zaber_positions):
            try:
                self.log_message(
                    f"Measurement {i + 1}: "
                    f"Zaber (x,y,z) = ({zaber_pos.x:.2f},{zaber_pos.y:.2f},{zaber_pos.z:.2f}) µm"
                )
                self.zaber.set_zaber_position_by_class(zaber_position=zaber_pos)

                frame = self.get_andor_frame()
                fitting = self.get_fitted_spectrum(frame)

                # Update GUI via provided callback
                display_results = self.get_display_results(frame, fitting)
                call_update_gui(display_results)

                measurement_point = MeasurementPoint(
                    frame=frame,
                    zaber_position=self.zaber.get_zaber_position_class(),
                    mako_image=None,
                )
                measurements.append(measurement_point)

            except Exception as e:
                self.log_message(f"[Measurement] Error at index {i}: {e}")

        return MeasurementSeries(
            measurements=measurements,
            state_mode=self.get_current_state_mode(),
            calibration_data=self.calibration_data,
            settings = measurement_settings,
        )



    def get_freq_shift(self, fitting: FittedSpectrum) -> float | None:
        return compute_freq_shift(fitting=fitting, calibration_calculator=self.calibration_calculator)


    def get_display_results(self, frame: np.ndarray, fitting: FittedSpectrum) -> DisplayResults:
        if self.is_reference_mode:
            freq_shift_ghz = self.microwave.get_frequency()
        elif fitting.is_success:
            freq_shift_ghz = self.get_freq_shift(fitting)
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
            try:
                frame = self._get_camera_snap()
            finally:
                # Always cancel the timer (even if _get_camera_snap() fails)
                timer.cancel()
        finally:
            # Always close the shutter at the end (even if an exception occurs)
            self.shutter_manager.sample.close()

        return frame

    def set_camera_settings(self,
                            exposure_time: float,
                            emccd_gain: int,
                            ):
        self.camera.set_exposure_time(exposure_time)
        self.camera.set_emccd_gain(emccd_gain)

        andor_config = andor_frame_config.get()


        self.camera.set_roi(x_start=andor_config.x_start,
                            x_end=andor_config.x_end,
                            y_start=andor_config.y_start,
                            y_end=andor_config.y_end,)
        self.camera.set_binning(hbin=andor_config.hbin,
                                vbin=andor_config.vbin)


        if self.is_reference_mode:
            self.reference_state_mode.camera_settings = self.get_andor_camera_settings()
        else:
            self.sample_state_mode.camera_settings = self.get_andor_camera_settings()





    def get_andor_camera_settings(self) -> AndorCameraSettings:
        return AndorCameraSettings(name=self.camera.get_name(),
                                   exposure_time_s=self.camera.get_exposure_time(),
                                   emccd_gain=self.camera.get_emccd_gain(),
                                   roi=self.camera.get_roi(),
                                   binning=self.camera.get_binning(),
                                   preamp_gain=self.camera.get_preamp_gain(),
                                   preamp_mode=f"{self.camera.get_amp_mode()}")
