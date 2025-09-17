import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Callable

import numpy as np


from brillouin_system.devices.cameras.andor.andor_frame.andor_config import andor_frame_config, AndorConfig
from brillouin_system.calibration.config.calibration_config import CalibrationConfig, calibration_config
from brillouin_system.devices.cameras.andor.baseCamera import BaseCamera
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
from brillouin_system.devices.microwave_device import Microwave, MicrowaveDummy
from brillouin_system.devices.shutter_device import ShutterManager, ShutterManagerDummy
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLensDummy
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterface, \
    ZaberHumanInterfaceDummy

from brillouin_system.my_dataclasses.background_image import ImageStatistics, generate_image_statistics_dataclass
from brillouin_system.my_dataclasses.display_results import DisplayResults
from brillouin_system.my_dataclasses.human_interface_measurements import RequestAxialScan, MeasurementPoint, AxialScan
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.calibration.calibration import CalibrationData, \
    CalibrationMeasurementPoint, MeasurementsPerFreq, CalibrationCalculator, CalibrationPolyfitParameters, calibrate
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum


from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.spectrum_fitting.helpers.subtract_background import subtract_background
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


class SystemType(Enum):
    HUMAN_INTERFACE = 0
    MICROSCOPE = 1

def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 using contrast stretching."""
    img = np.copy(image)
    img = np.clip(img, 0, np.percentile(img, 99))  # contrast stretch
    img = 255 * (img / img.max()) if img.max() > 0 else img
    return img.astype(np.uint8)



class HiBackend:

    @staticmethod
    def log_message(msg: str):
        """Optional helper to log messages — or use the signaller’s emit if needed."""
        print(f"[BrillouinBackend] {msg}")

    def __init__(self,
                 camera: BaseCamera | DummyCamera,
                 shutter_manager: ShutterManager | ShutterManagerDummy,
                 microwave: Microwave | MicrowaveDummy,
                 zaber_eye_lens: None | ZaberEyeLens | ZaberEyeLensDummy = None,
                 zaber_hi: None | ZaberHumanInterface | ZaberHumanInterfaceDummy = None,
                 is_sample_illumination_continuous: bool = False,
                 ):


        # init Spectrum Fitter:
        self.spectrum_fitter = SpectrumFitter()

        # Devices
        self.andor_camera: BaseCamera | DummyCamera = camera
        self.andor_camera.set_from_config_file(andor_frame_config.get())

        self.shutter_manager: ShutterManager | ShutterManagerDummy = shutter_manager

        self.microwave: Microwave | MicrowaveDummy = microwave

        self.microwave.set_power(power_dbm=-20)

        self.zaber_eye_lens = zaber_eye_lens
        self.zaber_hi = zaber_hi


        # State
        self.is_sample_illumination_continuous: bool = is_sample_illumination_continuous
        self.is_reference_mode: bool = False
        self.do_background_subtraction: bool = False
        self.do_live_fitting = False

        # Calibration
        self.calibration_data: CalibrationData | None = None
        self.calibration_poly_fit_params: CalibrationPolyfitParameters | None = None
        self.calibration_calculator: CalibrationCalculator | None = None

        # Background (BG) Image and dark_image for the sample
        self.bg_image: ImageStatistics | None = None
        self.dark_image: ImageStatistics | None = None

        self.init_shutters()
        self.init_camera_settings()

        # Init state modes
        self.reference_state_mode: SystemState = self.init_state_mode(is_reference_mode=True)
        self.sample_state_mode: SystemState = self.init_state_mode(is_reference_mode=False)


        # Init Signals (they are sent down from the signaller)
        # b2f = backend to frontend
        # f2b = frontend to backend
        self.b2f_send_system_state_signal = None
        self.b2f_emit_display_result = None
        self.f2b_cancel_callback = None

        # Init Zaber position Signals
        # Human Interface
        self.b2f_emit_update_zaber_lens_position = None



        # Store measurements for Human Interface
        self._i_axial_scans: int = 0
        self.axial_scan_dict: dict[int, AxialScan] = {}

    def get_list_of_axial_scans(self) -> list[str]:
        if not self.axial_scan_dict:
            return [""]
        lines = [f"{scan.i} - ID: {scan.id}" for scan in sorted(self.axial_scan_dict.values(), key=lambda s: s.i)]
        return lines

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

        self.andor_camera.set_pre_amp_mode(index=andor_config.pre_amp_mode)
        self.andor_camera.set_vss_index(index=andor_config.vss_index)

        self.andor_camera.set_roi(x_start=andor_config.x_start,
                                  x_end=andor_config.x_end,
                                  y_start=andor_config.y_start,
                                  y_end=andor_config.y_end, )
        self.andor_camera.set_binning(hbin=andor_config.hbin,
                                      vbin=andor_config.vbin)

        self.andor_camera.set_flip_image_horizontally(flip=andor_config.flip_image_horizontally)


    def init_state_mode(self, is_reference_mode: bool) -> SystemState:
        return SystemState(
            is_reference_mode=is_reference_mode,
            is_do_bg_subtraction_active=False,
            bg_image=None,
            dark_image=None,
            andor_camera_info=self.andor_camera.get_camera_info_dataclass()
        )



    def init_f2b_signals(self, cancel_callback: Callable[[], bool]):
        self.f2b_cancel_callback = cancel_callback

    def init_b2f_emit_display_result(self, emit_display_result: Callable[[DisplayResults], None]):
        self.b2f_emit_display_result = emit_display_result

    def init_b2f_zaber_position_updates_human_interface(self,
                                                        emit_update_zaber_lens_position:
                                                        Callable[[float], None]):
        self.b2f_emit_update_zaber_lens_position = emit_update_zaber_lens_position



    # ---------------- Change Modes ----------------

    def change_illumination_mode_to_continuous(self):
        self.is_sample_illumination_continuous = True
        self.shutter_manager.sample.open()
        print("[BrillouinBackend] Switched to continuous illumination mode.")

    def change_illumination_mode_to_pulsed(self):
        self.is_sample_illumination_continuous = False
        self.shutter_manager.sample.close()
        print("[BrillouinBackend] Switched to pulsed illumination mode.")

    def change_system_state(self, state_mode: SystemState):
        self.is_reference_mode = state_mode.is_reference_mode
        self.do_background_subtraction = state_mode.is_do_bg_subtraction_active
        self.bg_image = state_mode.bg_image
        self.set_andor_exposure(
            exposure_time=state_mode.andor_camera_info.exposure,
            emccd_gain=state_mode.andor_camera_info.gain,
        )

    def get_current_system_state(self) -> SystemState:
        return SystemState(
            is_reference_mode=self.is_reference_mode,
            is_do_bg_subtraction_active=self.do_background_subtraction,
            bg_image=self.bg_image,
            dark_image=self.dark_image,
            andor_camera_info=self.andor_camera.get_camera_info_dataclass()
        )

    def change_to_reference_mode(self):
        # Store current state mode of the sample for future:
        self.sample_state_mode = self.get_current_system_state()

        self.shutter_manager.change_to_reference()
        self.change_system_state(state_mode=self.reference_state_mode)
        print("[BrillouinBackend] Switched to reference mode.")


    def change_to_sample_mode(self):
        # Store current state mode of the sample for future:
        self.reference_state_mode = self.get_current_system_state()
        self.shutter_manager.change_to_objective()
        self.change_system_state(state_mode=self.sample_state_mode)
        print("[BrillouinBackend] Switched to sample mode.")



    # ----------------- Background Subtraction ----------------- #

    def start_background_subtraction(self):
        if self.is_background_image_available():
            self.do_background_subtraction = True
        else:
            self.do_background_subtraction = False
            print("[BrillouinBackend] No Background Image available")

    def stop_background_subtraction(self):
        self.do_background_subtraction = False

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        if not self.is_background_image_available():
            print("[AcquisitionManager] No background image available")
            return frame
        return subtract_background(frame=frame, bg_frame=self.bg_image)


    def take_n_images(self, n_images) -> np.ndarray:
        return np.stack([self._get_andor_camera_snap() for _ in range(n_images)])


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
        print(f"[BrillouinBackend] Taking {n_bg_images} Background Images...")
        n_images = self.take_n_images(n_bg_images)

        if isinstance(self.andor_camera, DummyCamera):
            n_images = n_images * 0.8


        print("[BrillouinBackend] ...Background Images acquired.")


        if self.is_sample_illumination_continuous:
            self.shutter_manager.sample.open()
        else:
            pass # do not open shutter, we are in snap mode

        return generate_image_statistics_dataclass(n_images)


    def get_dark_image(self) -> ImageStatistics | None:
        andor_config = andor_frame_config.get()

        n_dark_images = andor_config.n_dark_images

        if n_dark_images == 0:
            print(
                f"[BrillouinBackend] No Dark Images Requested")
            return None

        # Info:
        settings = self.andor_camera.get_exposure_dataclass()
        print(
            f"[BrillouinBackend] Acquired {n_dark_images} dark images | Exposure: "
            f"{settings.exposure_time_s:.3f}s | EM Gain: {settings.emccd_gain}")

        self.andor_camera.close_shutter()
        time.sleep(0.1)

        n_images = self.take_n_images(n_dark_images)

        if isinstance(self.andor_camera, DummyCamera):
            n_images = n_images * 0.01

        self.andor_camera.open_shutter()
        time.sleep(0.05)

        print(f"[BrillouinBackend] {n_dark_images} dark images acquired with: {self.andor_camera.get_exposure_dataclass()}")

        return generate_image_statistics_dataclass(n_images)



    def is_background_image_available(self) -> bool:
        if self.bg_image is None:
            return False
        else:
            return True


    # ---------------- Get Frames  ----------------
    def _get_andor_camera_snap(self) -> np.ndarray:
        """Pull a raw frame from the camera."""
        return self.andor_camera.snap().astype(np.float64)


    def get_andor_frame(self) -> np.ndarray:
        # Get the frame:
        if self.is_reference_mode or self.is_sample_illumination_continuous:
            frame = self._get_andor_camera_snap()
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
            px, sline = self.spectrum_fitter.get_px_sline_from_image(frame_with_sub_bg)
        else:
            px, sline = self.spectrum_fitter.get_px_sline_from_image(frame)

        if not self.do_live_fitting and not self.is_reference_mode:
            return self.spectrum_fitter.get_empty_fitting(px, sline)

        try:
            return self.spectrum_fitter.fit(px, sline, is_reference_mode=self.is_reference_mode)
        except Exception as e:
            print(f"[BrillouinBackend] Fitting error: {e}")
            return self.spectrum_fitter.get_empty_fitting(px, sline)

    def update_calibration_calculator(self):
        if self.calibration_data is None:
            self.calibration_poly_fit_params = None
            self.calibration_calculator = None
        else:
            self.calibration_poly_fit_params = calibrate(data=self.calibration_data)
            self.calibration_calculator: CalibrationCalculator = CalibrationCalculator(parameters=self.calibration_poly_fit_params)


    def take_one_measurement(self) -> MeasurementPoint:


        return MeasurementPoint(
            frame_andor=self.get_andor_frame(),
            lens_zaber_position=self.zaber_eye_lens.get_position(),
            frame_left_allied=None,
            frame_right_allied=None,
        )

    def take_axial_scan(self, request_axial_scan: RequestAxialScan):

        if self.is_reference_mode:
            print(f"[Axial Scan] Measuring N Times the Reference Signal {request_axial_scan.n_measurements}.")
            lens_x0 = self.zaber_eye_lens.get_position()

            all_results = []

            for i in range(request_axial_scan.n_measurements):
                if self.f2b_cancel_callback():
                    print(f"[Axial Scan] Cancelled during step {i}.")
                    return False

                frame = self._get_andor_camera_snap()

                fs = self.get_fitted_spectrum(frame)
                self.b2f_emit_display_result(self.get_display_results(frame=frame, fitting=fs))

                all_results.append(
                    MeasurementPoint(
                    frame_andor=frame,
                    lens_zaber_position=lens_x0,
                    frame_left_allied=None,
                    frame_right_allied=None,)
                )

        else:
            lens_x0 = self.zaber_eye_lens.get_position()
            dx = request_axial_scan.step_size_um

            print(f"[Axial Scan] Starting: {request_axial_scan.n_measurements} steps, "
                  f"step size: {request_axial_scan.step_size_um} µm, "
                  f"ID: {request_axial_scan.id}")

            all_results = []

            try:
                if not self.is_sample_illumination_continuous:
                    self.shutter_manager.sample.open()

                for i in range(request_axial_scan.n_measurements):
                    if self.f2b_cancel_callback():
                        print(f"[Axial Scan] Cancelled during step {i}. Returning lens to starting position.")
                        self.zaber_eye_lens.move_abs(lens_x0)
                        return False

                    print(f"[Axial Scan] Frame {i}/{request_axial_scan.n_measurements}")
                    self.zaber_eye_lens.move_rel(dx)
                    zaber_pos = self.zaber_eye_lens.get_position()
                    self.b2f_emit_update_zaber_lens_position(zaber_pos)

                    frame = self._get_andor_camera_snap()

                    fs = self.get_fitted_spectrum(frame)
                    self.b2f_emit_display_result(self.get_display_results(frame=frame, fitting=fs))

                    all_results.append(
                        MeasurementPoint(
                        frame_andor=frame,
                        lens_zaber_position=zaber_pos,
                        frame_left_allied=None,
                        frame_right_allied=None,)
                    )

            finally:
                if not self.is_sample_illumination_continuous:
                    self.shutter_manager.sample.close()

        self._i_axial_scans += 1

        axial_scan = AxialScan(
            i=self._i_axial_scans,
            id=request_axial_scan.id,
            measurements=all_results,
            system_state=self.get_current_system_state(),
            calibration_params=self.calibration_poly_fit_params,
            eye_location= None
        )
        self.axial_scan_dict[axial_scan.i] = axial_scan

    def get_axial_scan_data(self, index: int):
        try:
            return self.axial_scan_dict[index]
        except (IndexError, KeyError):
            return None

    def get_freq_shift(self, fitting: FittedSpectrum) -> float | None:
        if self.calibration_calculator is None:
            return None
        else:
            return self.calibration_calculator.compute_freq_shift(fitting=fitting)


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
                mask_for_fitting=fitting.mask_for_fitting,
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
                frame = self._get_andor_camera_snap()
            finally:
                # Always cancel the timer (even if _get_camera_snap() fails)
                timer.cancel()
        finally:
            # Always close the shutter at the end (even if an exception occurs)
            self.shutter_manager.sample.close()

        return frame

    def set_andor_exposure(self,
                            exposure_time: float,
                            emccd_gain: int,
                            ):

        self.andor_camera.set_exposure_time(seconds=exposure_time)
        self.andor_camera.set_emccd_gain(gain=emccd_gain)


        if self.is_reference_mode:
            self.reference_state_mode.andor_camera_info.exposure = exposure_time
            self.reference_state_mode.andor_camera_info.gain = emccd_gain
        else:
            self.sample_state_mode.andor_camera_info.exposure = exposure_time
            self.sample_state_mode.andor_camera_info.gain = emccd_gain


    def update_andor_config_settings(self, andor_config: AndorConfig):
        self.andor_camera.set_from_config_file(andor_config)

    @contextmanager
    def force_reference_mode(self):
        was_sample_mode = not self.is_reference_mode
        if was_sample_mode:
            self.change_to_reference_mode()
        try:
            yield
        finally:
            if was_sample_mode:
                self.change_to_sample_mode()


    def perform_calibration(self) -> bool:

        config: CalibrationConfig = calibration_config.get()

        print("[Calibration] Starting calibration.")

        try:
            with self.force_reference_mode():
                measured_freqs = []

                for freq in config.calibration_freqs:
                    if self.f2b_cancel_callback():
                        print("[Calibration] Cancelled by user.")
                        return False

                    self.microwave.set_frequency(freq)
                    freq_points = []

                    for _ in range(config.n_per_freq):
                        if self.f2b_cancel_callback():
                            print("[Calibration] Cancelled by user.")
                            return False

                        frame = self.get_andor_frame()
                        fs = self.get_fitted_spectrum(frame)

                        cali_point = CalibrationMeasurementPoint(
                            frame=frame,
                            microwave_freq=self.microwave.get_frequency(),
                            fitting_results=fs,
                        )
                        freq_points.append(cali_point)
                        self.b2f_emit_display_result(self.get_display_results(frame, fs))

                    measured_freqs.append(MeasurementsPerFreq(
                        set_freq_ghz=freq,
                        state_mode=self.get_current_system_state(),
                        cali_meas_points=freq_points
                    ))

                self.calibration_data = CalibrationData(measured_freqs=measured_freqs)
                self.update_calibration_calculator()
                print("[Calibration] Completed successfully.")
                return True

        except Exception as e:
            print(f"[Calibration] Exception: {e}")
            return False



    def close(self):
        """Cleanly shut down all backend-controlled devices."""
        self.log_message("Shutting down BrillouinBackend devices...")

        try:
            self.shutter_manager.close_all()
            self.log_message("Shutters closed.")
        except Exception as e:
            self.log_message(f"Error closing shutter manager: {e}")

        try:
            self.andor_camera.close()
            self.log_message("Andor camera closed.")
        except Exception as e:
            self.log_message(f"Error closing Andor camera: {e}")

        try:
            self.microwave.shutdown()
            self.log_message("Microwave shut down.")
        except Exception as e:
            self.log_message(f"Error shutting down microwave: {e}")

        try:
            self.zaber_eye_lens.close()
            self.log_message("Zaber controller closed.")
        except Exception as e:
            self.log_message(f"Error closing Zaber controller: {e}")


        self.log_message("BrillouinBackend shutdown complete.")

