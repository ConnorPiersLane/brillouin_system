
import time
from contextlib import contextmanager
from enum import Enum
from typing import Callable

import numpy as np


from brillouin_system.devices.cameras.andor.andor_frame.andor_config import andor_frame_config, AndorConfig
from brillouin_system.calibration.config.calibration_config import CalibrationConfig, calibration_config
from brillouin_system.devices.cameras.andor.baseCamera import BaseCamera
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
from brillouin_system.devices.microwave_device import Microwave, MicrowaveDummy

from brillouin_system.devices.shutter_device import ShutterManager, ShutterManagerDummy
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens_dummy import ZaberEyeLensDummy
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterface, \
    ZaberHumanInterfaceDummy
from brillouin_system.eye_tracker.calibrate_camera_laser_position.calib_rig_laser_position import LaserOffset, \
    CalibRigLaserPosition

from brillouin_system.my_dataclasses.my_exceptions import OperationCancelled
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult, find_reflection_realtime
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig, \
    axial_scanning_config
from brillouin_system.logging_utils.logging_setup import get_logger

from brillouin_system.my_dataclasses.background_image import ImageStatistics, generate_image_statistics_dataclass
from brillouin_system.my_dataclasses.display_results import DisplayResults
from brillouin_system.my_dataclasses.human_interface_measurements import RequestAxialStepScan, MeasurementPoint, \
    AxialScan
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.calibration.calibration import CalibrationData, \
    CalibrationMeasurementPoint, MeasurementsPerFreq, CalibrationCalculator, CalibrationPolyfitParameters, calibrate
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.spectrum_fitting.helpers.subtract_background import subtract_background
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


log = get_logger(__name__)


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


    def __init__(self,
                 use_dummy: bool = True
                 ):


        # init Spectrum Fitter:
        self.spectrum_fitter = SpectrumFitter()

        # Devices
        if use_dummy:
            camera=DummyCamera()
            shutter_manager=ShutterManagerDummy('human_interface')
            microwave=MicrowaveDummy()
            zaber_eye_lens=ZaberEyeLensDummy()
            zaber_hi=ZaberHumanInterfaceDummy()
            # zaber_eye_lens=ZaberEyeLens()
            # zaber_hi=ZaberHumanInterface()
            from brillouin_system.devices.ni.ni_dummy import NIDummy
            ni = NIDummy()

        else:
            try:
                camera=IxonUltra(
                    index = 0,
                    temperature = "off",
                    fan_mode = "full",
                    x_start = 40, x_end  = 120,
                    y_start= 300, y_end  = 315,
                    vbin= 1, hbin  = 1,
                    verbose = True,
                    advanced_gain_option=False
                )
                # camera = DummyCamera()
            except:
                raise print("Camera not connected")

            shutter_manager=ShutterManager('human_interface')
            # shutter_manager = ShutterManager('microscope')
            try:
                microwave=Microwave()
                # microwave = MicrowaveDummy()
            except:
                raise print("Microwave not connected")
            zaber_eye_lens=ZaberEyeLens()
            # zaber_eye_lens = ZaberEyeLensDummy()
            zaber_hi=ZaberHumanInterface()
            # zaber_hi = ZaberHumanInterfaceDummy()
            from brillouin_system.devices.ni.ni6008 import NI6008
            ni = NI6008()

        self.andor_camera: BaseCamera | DummyCamera | IxonUltra = camera
        self._andor_config: AndorConfig = andor_frame_config.get()
        self.update_andor_config_settings(andor_config=self._andor_config)

        self._axial_scan_config: ScanningConfig = axial_scanning_config.get()

        self.shutter_manager: ShutterManager | ShutterManagerDummy = shutter_manager

        self.microwave: Microwave | MicrowaveDummy = microwave

        self.microwave.set_power(power_dbm=-20)

        self.zaber_eye_lens = zaber_eye_lens
        self.zaber_hi = zaber_hi

        # DAQ
        self.ni = ni

        # State
        self.is_shutter_open: bool = True
        self.is_reference_mode: bool = False
        self.do_background_subtraction: bool = False
        self.do_live_fitting = False

        # Calibration
        self.calibration_data: CalibrationData | None = None
        self.calibration_poly_fit_params: CalibrationPolyfitParameters | None = None
        self.calibration_calculator: CalibrationCalculator | None = None
        self.calibration_config: CalibrationConfig = calibration_config.get()

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
        self.f2b_cancel_callback: Callable[[], bool] = lambda: False

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
        if self.is_shutter_open:
            self.shutter_manager.sample.open()
        else:
            self.shutter_manager.sample.close()

    def init_camera_settings(self):
        andor_config = self._andor_config

        self.andor_camera.set_fixed_pre_amp_mode(index=andor_config.pre_amp_mode)
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

    def update_calibration_config(self, config: CalibrationConfig):
        self.calibration_config = config

    def init_f2b_signals(self, cancel_callback: Callable[[], bool]):
        self.f2b_cancel_callback = cancel_callback

    def init_b2f_emit_display_result(self, emit_display_result: Callable[[DisplayResults], None]):
        self.b2f_emit_display_result = emit_display_result

    def init_b2f_zaber_position_updates_human_interface(self,
                                                        emit_update_zaber_lens_position:
                                                        Callable[[float], None]):
        self.b2f_emit_update_zaber_lens_position = emit_update_zaber_lens_position

    def move_and_update_gui_zaber_eye_lens_rel(self, dz_um: float) -> float:
        """
        Move Zaber eye lens by a relative distance (µm).

        Returns:
            New absolute lens position (µm).
        """
        self.zaber_eye_lens.move_rel(dz_um)
        z = self.zaber_eye_lens.get_position()
        if self.b2f_emit_update_zaber_lens_position:
            self.b2f_emit_update_zaber_lens_position(z)
        return z

    def move_and_update_gui_zaber_eye_lens_abs(self, z_um: float) -> float:
        """
        Move Zaber eye lens to an absolute position (µm).

        Returns:
            New absolute lens position (µm).
        """
        self.zaber_eye_lens.move_abs(z_um)
        z = self.zaber_eye_lens.get_position()
        if self.b2f_emit_update_zaber_lens_position:
            self.b2f_emit_update_zaber_lens_position(z)
        return z

    # ---------------- Change Modes ----------------



    def open_sample_shutter(self):
        self.is_shutter_open = True
        self.shutter_manager.objective.open()
        log.info("[BrillouinBackend] Switched to continuous illumination mode.")

    def close_sample_shutter(self):
        self.is_shutter_open = False
        self.shutter_manager.objective.close()
        log.info("[BrillouinBackend] Switched to pulsed illumination mode.")

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
        log.info("[BrillouinBackend] Switched to reference mode.")


    def change_to_sample_mode(self):
        # Store current state mode of the sample for future:
        self.reference_state_mode = self.get_current_system_state()
        self.shutter_manager.reference.close()
        if self.is_shutter_open:
            self.shutter_manager.objective.open()
        self.change_system_state(state_mode=self.sample_state_mode)
        log.info("[BrillouinBackend] Switched to sample mode.")



    # ----------------- Background Subtraction ----------------- #

    def start_background_subtraction(self):
        if self.is_background_image_available():
            self.do_background_subtraction = True
        else:
            self.do_background_subtraction = False
            log.info("[BrillouinBackend] No Background Image available")

    def stop_background_subtraction(self):
        self.do_background_subtraction = False
        log.info("[BrillouinBackend] Background subtraction unabled")

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        if not self.is_background_image_available():
            log.info("[AcquisitionManager] No background image available")
            return frame
        return subtract_background(frame=frame, bg_frame=self.bg_image)

    def take_n_images(self, n_images: int) -> np.ndarray:
        """Acquire up to n_images, with cancel support and progress logging.

        Cancellation is checked between snaps, so Stop/Cancel can interrupt the
        sequence cleanly. The current in-flight camera snap cannot be interrupted,
        but acquisition stops before the next frame.
        """
        frames: list[np.ndarray] = []

        if n_images <= 0:
            log.info("[Acquisition] No images requested.")
            return np.empty((0,), dtype=np.float64)

        log.info(f"[Acquisition] Starting acquisition of {n_images} image(s).")

        for i in range(n_images):
            if self.f2b_cancel_callback():
                log.info(f"[Acquisition] Cancelled at {i}/{n_images} image(s).")
                raise OperationCancelled()

            frame = self._get_andor_camera_snap()
            frames.append(frame)

            log.info(f"[Acquisition] Progress: {i + 1}/{n_images}")

        if not frames:
            log.warning("[Acquisition] No images acquired.")
            return np.empty((0,), dtype=np.float64)

        log.info(f"[Acquisition] Finished with {len(frames)}/{n_images} image(s) acquired.")
        return np.stack(frames, axis=0)


    def take_bg_and_darknoise_images(self):

        self.dark_image: ImageStatistics = self.get_dark_image(n_images=self._andor_config.n_dark_images)
        self.bg_image: ImageStatistics = self.get_bg_image(n_images=self._andor_config.n_bg_images)




    def get_bg_image(self, n_images: int) -> ImageStatistics:
        """Capture and average multiple frames to use as background."""

        if self.is_shutter_open:
            self.shutter_manager.sample.close()
        else:
            pass # shutter should already be closed
        time.sleep(0.05)  # Optional delay before acquisition

        # andor_config = self._andor_config

        log.info(f"Taking {n_images} Background Images...")
        n_images = self.take_n_images(n_images)

        if isinstance(self.andor_camera, DummyCamera):
            n_images = n_images * 0.8


        log.info("[BrillouinBackend] ...Background Images acquired.")


        if self.is_shutter_open:
            self.shutter_manager.sample.open()
        else:
            pass # do not open shutter

        return generate_image_statistics_dataclass(n_images)


    def get_dark_image(self, n_images: int) -> ImageStatistics | None:

        n_dark_images = n_images

        if n_dark_images == 0:
            log.info("No Dark Images Requested")
            return None

        # Info:
        self.andor_camera.close_shutter()
        time.sleep(0.1)

        n_images = self.take_n_images(n_dark_images)

        if isinstance(self.andor_camera, DummyCamera):
            n_images = n_images * 0.01

        self.andor_camera.open_shutter()
        time.sleep(0.05)

        log.info(f"{n_dark_images} dark images acquired with: {self.andor_camera.get_exposure_dataclass()}")

        return generate_image_statistics_dataclass(n_images)



    def is_background_image_available(self) -> bool:
        if self.bg_image is None:
            return False
        else:
            return True




    # ---------------- Get Frames  ----------------
    def _get_andor_camera_snap(self) -> np.ndarray:
        """Pull a raw frame from the camera.
        Returns: frame, time.time()
        """
        frame = self.andor_camera.snap()
        return frame.astype(np.float64)


    def get_andor_frame(self) -> np.ndarray:
        return self._get_andor_camera_snap()

    def get_fitted_spectrum(self, frame) -> FittedSpectrum:
        """
        Fits a Brillouin spectrum depending on reference mode and background subtraction.
        If live fitting is disabled, returns an unsuccessful fit but includes a raw spectrum line.

        Args:
            frame (np.ndarray): The input camera frame.

        Returns:
            FittedSpectrum: Dataclass containing fit results and metadata.
        """


        px, sline = self.spectrum_fitter.get_px_sline_from_image(frame)

        if not self.do_live_fitting and not self.is_reference_mode:
            return self.spectrum_fitter.get_empty_fitting(px, sline)

        try:
            anchors = (self.calibration_calculator.elastic_anchors()
                       if self.calibration_calculator is not None else None)
            return self.spectrum_fitter.fit(px, sline, is_reference_mode=self.is_reference_mode, anchors=anchors)
        except Exception as e:
            # A raised fit (as opposed to an unsuccessful FittedSpectrum) means
            # a real misconfiguration — e.g. a *_psf sample model selected
            # while the calibration has no PSF chain. Warn loudly so it is not
            # mistaken for an ordinary no-peak frame; no fit is shown.
            log.warning(f"Fitting skipped (no result): {e}")
            return self.spectrum_fitter.get_empty_fitting(px, sline)

    def update_calibration_calculator(self):
        if self.calibration_data is None:
            self.calibration_poly_fit_params = None
            self.calibration_calculator = None
        else:
            self.calibration_poly_fit_params = calibrate(data=self.calibration_data,
                                                         poyfit_degree=self.calibration_config.degree)
            self.calibration_calculator: CalibrationCalculator = CalibrationCalculator(
                parameters=self.calibration_poly_fit_params)




    def take_axial_step_scan(self, request_axial_scan: RequestAxialStepScan) -> bool:

        lens_x0 = self.zaber_eye_lens.get_position()
        all_results = []
        reflection_result_forwards: ReflectionResult | None = None
        reflection_result_backwards: ReflectionResult | None = None

        if self.is_reference_mode:
            log.info(f"[Axial Scan] Measuring N Times the Reference Signal {request_axial_scan.n_measurements}.")

            for i in range(request_axial_scan.n_measurements):
                log.info(f"[Axial Scan] Frame {i+1}/{request_axial_scan.n_measurements}")
                if self.f2b_cancel_callback():
                    log.info(f"[Axial Scan] Cancelled during step {i+1}.")
                    return False

                frame = self._get_andor_camera_snap()

                self.display_spectrum(frame=frame)

                all_results.append(
                    MeasurementPoint(
                    frame_andor=frame,
                    lens_zaber_position=lens_x0,
                    time_stamp=time.perf_counter())
                )

        else:

            dx = request_axial_scan.step_size_um

            log.info(f"[Axial Scan] Starting: {request_axial_scan.n_measurements} steps, "
                  f"step size: {request_axial_scan.step_size_um} µm, "
                  f"ID: {request_axial_scan.id}")

            if request_axial_scan.find_reflection_plane:
                reflection_result_forwards: ReflectionResult = self.find_reflection_plane(is_go_forwards=True)
                if reflection_result_forwards.found:
                    z_pos = reflection_result_forwards.event_z_um + reflection_result_forwards.z_offset_um
                    self.zaber_eye_lens.move_abs(z_pos)
                else:
                    self.zaber_eye_lens.move_abs(lens_x0)
                    return False

            for i in range(request_axial_scan.n_measurements):
                if self.f2b_cancel_callback():
                    log.info(f"[Axial Scan] Cancelled during step {i+1}. Returning lens to starting position.")
                    self.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
                    return False

                log.info(f"[Axial Scan] Frame {i+1}/{request_axial_scan.n_measurements}")
                self.zaber_eye_lens.move_rel(delta_um=dx)
                zaber_pos = self.zaber_eye_lens.get_position()
                self.b2f_emit_update_zaber_lens_position(zaber_pos)

                frame = self._get_andor_camera_snap()

                self.display_spectrum(frame=frame)

                all_results.append(
                    MeasurementPoint(
                        frame_andor=frame,
                        lens_zaber_position=zaber_pos,
                        time_stamp=time.perf_counter())
                )


        if request_axial_scan.find_reflection_plane:
            reflection_result_backwards: ReflectionResult = self.find_reflection_plane(is_go_forwards=False)


        # Move lens back to original position
        self.move_and_update_gui_zaber_eye_lens_abs(lens_x0)

        self._i_axial_scans += 1

        store_frames = self.calibration_config.save_calibration_frames
        axial_scan = AxialScan(
            i=self._i_axial_scans,
            id=request_axial_scan.id,
            measurements=all_results,
            system_state=self.get_current_system_state(),
            calibration_params=self.calibration_poly_fit_params,
            eye_tracker_results=request_axial_scan.eye_tracker_results,
            reflection_result_forwards=reflection_result_forwards,
            reflection_result_backwards=reflection_result_backwards,
            calibration_data=self.calibration_data if store_frames else None,
        )
        self.axial_scan_dict[axial_scan.i] = axial_scan

        return True


    def display_spectrum(self, frame):
        if self.do_background_subtraction:
            frame_with_sub_bg = self.subtract_background(frame)
            fs = self.get_fitted_spectrum(frame_with_sub_bg)
            self.b2f_emit_display_result(self.get_display_results(frame=frame_with_sub_bg, fitting=fs))
        else:
            fs = self.get_fitted_spectrum(frame)
            self.b2f_emit_display_result(self.get_display_results(frame=frame, fitting=fs))


    def get_axial_scan_data(self, index: int):
        try:
            return self.axial_scan_dict[index]
        except (IndexError, KeyError):
            return None

    def get_freq_shift(self, fitting: FittedSpectrum) -> float | None:
        if self.calibration_calculator is None:
            return None
        else:
            return self.calibration_calculator.compute_freq_shift(fitting=fitting,
                                                                  reference=self.calibration_config.reference,
                                                                  mode=self.calibration_config.mode)

    def get_hwhm_shift(self, fitting: FittedSpectrum) -> tuple:
        """

        Args:
            fitting:

        Returns: hwhm_left_peak_ghz, hwhm_right_peak_ghz

        """
        if self.calibration_calculator is None:
            return None, None
        else:
            calc = self.calibration_calculator.for_chain(fitting.calibration_chain)
            hwhm_left_peak_ghz = float(
                abs(
                    calc.df_left_peak(
                        fitting.left_peak_center_px,
                        fitting.left_peak_width_px
                    )))
            hwhm_right_peak_ghz = float(
                abs(
                    calc.df_right_peak(
                        fitting.right_peak_center_px,
                        fitting.right_peak_width_px
                    )))

            return hwhm_left_peak_ghz, hwhm_right_peak_ghz


    def get_display_results(self, frame: np.ndarray, fitting: FittedSpectrum) -> DisplayResults:
        if self.is_reference_mode:
            freq_shift_ghz = self.microwave.get_frequency()
            hwhm_lp_ghz, hwhm_rp_ghz = self.get_hwhm_shift(fitting)

        elif fitting.is_success:
            freq_shift_ghz = self.get_freq_shift(fitting)
            hwhm_lp_ghz, hwhm_rp_ghz = self.get_hwhm_shift(fitting)
        else:
            freq_shift_ghz = None
            hwhm_lp_ghz, hwhm_rp_ghz = None, None

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
                hwhm_left_peak=hwhm_lp_ghz,
                hwhm_right_peak=hwhm_rp_ghz,
            )
        else:
            return DisplayResults(
                is_fitting_available=False,
                frame=frame,
                x_pixels=fitting.x_pixels,
                sline=fitting.sline,
            )

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
        self._andor_config = andor_config

    def update_scanning_config_file(self, axial_scan_config: ScanningConfig):
        self._axial_scan_config = axial_scan_config

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

        log.info("[Calibration] Starting calibration.")

        try:
            with self.force_reference_mode():
                measured_freqs = []

                i = 0
                n = len(config.calibration_freqs)
                for freq in config.calibration_freqs:
                    if self.f2b_cancel_callback():
                        log.info("[Calibration] Cancelled by user.")
                        return False

                    self.microwave.set_frequency(freq)
                    i += 1
                    log.info(f"Freq {i}/{n}")
                    freq_points = []

                    for _ in range(config.n_per_freq):
                        if self.f2b_cancel_callback():
                            log.info("[Calibration] Cancelled by user.")
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
                log.info(self.calibration_calculator.get_str_all_models())
                return True

        except Exception as e:
            log.info(f"[Calibration] Exception: {e}")
            return False

    def find_reflection_plane(self, is_go_forwards: bool=True) -> ReflectionResult:
        """

        Args:

            is_go_forwards: True (forwards) False (backwards)

        Returns:

        """
        if self.is_reference_mode:
            log.info(f"System is in Reference (Calibration Mode) - Change to Sample Mode")
            return ReflectionResult(found=False)

        ni_sample_rate_hz = self._axial_scan_config.ni_sample_rate_hz
        if is_go_forwards:
            speed_um_s = self._axial_scan_config.speed_um_s
        else:
            speed_um_s = -self._axial_scan_config.speed_um_s
        max_distance_um = self._axial_scan_config.max_distance_um
        threshold_high_n_sigma = self._axial_scan_config.threshold_high_n_sigma
        threshold_low_n_sigma = self._axial_scan_config.threshold_low_n_sigma
        bg_acqui_s = self._axial_scan_config.bg_acqui_s
        debounce_s = self._axial_scan_config.debounce_s
        z_poll_s = self._axial_scan_config.z_poll_s
        chunk_size = self._axial_scan_config.chunk_size
        idle_sleep_s = self._axial_scan_config.idle_sleep_s
        offset_z_um = self._axial_scan_config.z_offset_um
        result: ReflectionResult = find_reflection_realtime(
            ni=self.ni,
            zaber=self.zaber_eye_lens,
            ni_sample_rate_hz=ni_sample_rate_hz,
            speed_um_s=speed_um_s,
            max_distance_um=max_distance_um,
            threshold_high_n_sigma=threshold_high_n_sigma,
            threshold_low_n_sigma=threshold_low_n_sigma,
            bg_acqui_s=bg_acqui_s,
            debounce_s=debounce_s,
            z_poll_s=z_poll_s,
            chunk_size=chunk_size,
            idle_sleep_s=idle_sleep_s,
            z_offset_um=offset_z_um,
        )
        return result


    def run_laser_xy_calibration(self) -> LaserOffset:
        """
        Run the full laser XY calibration from the backend and save offset.toml.

        Returns:
            LaserCoordSystem
        """
        if self.is_reference_mode:
            raise RuntimeError("Laser XY calibration must be run in sample mode, not reference mode.")

        log.info("[Laser XY Calibration] Starting.")

        calib = CalibRigLaserPosition(
            ni=self.ni,
            zaber_eye_lens=self.zaber_eye_lens,
            zaber_hi=self.zaber_hi,
            cancel_callback=self.f2b_cancel_callback,
            axial_scan_config=self._axial_scan_config,
        )

        try:
            laser_coord_system = calib.run_calibration()

            log.info(
                f"[Laser XY Calibration] Done. "
                f"dx={laser_coord_system.dx:.3f}, "
                f"dy={laser_coord_system.dy:.3f}, "
                f"dz={laser_coord_system.dz:.3f}"
            )
            return laser_coord_system

        except Exception as e:
            log.exception(f"[Laser XY Calibration] Failed: {e}")
            raise


    def close(self):
        """Cleanly shut down all backend-controlled devices."""
        print("Shutting down BrillouinBackend devices...")

        try:
            self.shutter_manager.close_all()
            print("Shutters closed.")
        except Exception as e:
            print(f"Error closing shutter manager: {e}")

        try:
            self.andor_camera.close()
            print("Andor camera closed.")
        except Exception as e:
            print(f"Error closing Andor camera: {e}")

        try:
            self.microwave.shutdown()
            print("Microwave shut down.")
        except Exception as e:
            print(f"Error shutting down microwave: {e}")

        try:
            self.zaber_eye_lens.close()
            print("Zaber controller closed.")
        except Exception as e:
            print(f"Error closing Zaber controller: {e}")


        print("BrillouinBackend shutdown complete.")

