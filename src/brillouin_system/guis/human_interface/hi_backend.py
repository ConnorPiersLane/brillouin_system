
import time
from contextlib import contextmanager
from typing import Callable

import numpy as np


from brillouin_system.devices.cameras.andor.andor_frame.andor_config import andor_frame_config, AndorConfig
from brillouin_system.calibration.config.calibration_config import CalibrationConfig, calibration_config
from brillouin_system.devices.cameras.andor.baseCamera import BaseCamera
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
from brillouin_system.devices.microwave_device import Microwave, MicrowaveDummy

from brillouin_system.devices.shutter_device import ShutterManager, ShutterManagerDummy
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterface, \
    ZaberHumanInterfaceDummy
from brillouin_system.eye_tracker.calibrate_camera_laser_position.calib_rig_laser_position import LaserOffset, \
    CalibRigLaserPosition

from brillouin_system.my_dataclasses.my_exceptions import OperationCancelled
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult, find_reflection_realtime
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig, \
    axial_scanning_config
from brillouin_system.scan_managers.sweep_scan_config.sweep_scan_config import SweepScanConfig, \
    sweep_scan_config
from brillouin_system.logging_utils.logging_setup import get_logger

from brillouin_system.my_dataclasses.background_image import ImageStatistics, generate_image_statistics_dataclass
from brillouin_system.my_dataclasses.display_results import DisplayResults
from brillouin_system.my_dataclasses.human_interface_measurements import RequestAxialStepScan, RequestSweepScan, \
    MeasurementPoint, AxialScan, SweepCycle
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.calibration.calibration import CalibrationData, \
    CalibrationMeasurementPoint, MeasurementsPerFreq, CalibrationCalculator, CalibrationPolyfitParameters, calibrate
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.spectrum_fitting.helpers.subtract_background import subtract_background
from brillouin_system.spectrum_fitting.elastic_anchors import ElasticAnchors
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter, model_requires_anchors, \
    config_requires_reflection_background
from brillouin_system.spectrum_fitting.reflection_background import (
    ReflectionBackground,
    ReflectionBackgroundMapper,
)


log = get_logger(__name__)


class HiBackend:


    def __init__(self,
                 use_dummy: bool = True
                 ):


        # init Spectrum Fitter:
        self.spectrum_fitter = SpectrumFitter()

        # Reflection background ("ReflectionBG") for the reflection
        # background / prmr preset: the packaged template is loaded lazily,
        # the mapper is rebuilt whenever the calibration or row band changes.
        self._reflection_bg: ReflectionBackground | None = None
        self._reflection_mapper: ReflectionBackgroundMapper | None = None
        self._reflection_mapper_key = None

        # Devices
        if use_dummy:
            camera=DummyCamera()
            shutter_manager=ShutterManagerDummy('human_interface')
            microwave=MicrowaveDummy()
            # Simulated NI + eye lens with a moving simulated cornea: the DAQ
            # signal is coupled to the lens-cornea distance, so reflection
            # finding actually works in dummy mode.
            from brillouin_system.patient_movement_analysis.simulated_devices import (
                SimNI, SimZaberLens, SimulatedCornea)
            sim_cornea = SimulatedCornea()
            zaber_eye_lens = SimZaberLens(start_um=9000.0)
            ni = SimNI(zaber_eye_lens, sim_cornea)
            zaber_hi=ZaberHumanInterfaceDummy()

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
            except Exception as e:
                raise RuntimeError("Camera not connected") from e

            shutter_manager=ShutterManager('human_interface')
            try:
                microwave=Microwave()
                # microwave = MicrowaveDummy()
            except Exception as e:
                raise RuntimeError("Microwave not connected") from e
            zaber_eye_lens=ZaberEyeLens()
            zaber_hi=ZaberHumanInterface()
            from brillouin_system.devices.ni.ni6008 import NI6008
            ni = NI6008()

        self.andor_camera: BaseCamera | DummyCamera | IxonUltra = camera
        self._andor_config: AndorConfig = andor_frame_config.get()
        self.update_andor_config_settings(andor_config=self._andor_config)

        self._axial_scan_config: ScanningConfig = axial_scanning_config.get()
        self._sweep_scan_config: SweepScanConfig = sweep_scan_config.get()

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
            anchors = self._elastic_anchors_if_required()
            measured_bg = self._reflection_background_if_required(px)
            return self.spectrum_fitter.fit(px, sline, is_reference_mode=self.is_reference_mode,
                                            anchors=anchors,
                                            measured_background=measured_bg)
        except Exception as e:
            log.info(f"Fitting error: {e}")
            return self.spectrum_fitter.get_empty_fitting(px, sline)

    def _calibration_data_to_store(self) -> CalibrationData | None:
        """The raw calibration frames to travel with a scan, or None.

        Re-fitting a scan against its OWN calibration is only possible if the
        frames are stored with it; the fitted polynomials alone cannot be
        re-derived with a different lineshape model. Disable via
        calibration_config.save_calibration_frames to save disk space.
        """
        if not self.calibration_config.save_calibration_frames:
            return None
        return self.calibration_data

    def _sline_rows_for_scan(self, measurements) -> list[int] | None:
        """Rows summed into the spectral line, to be stored with the scan.

        In automatic mode the band is located once (from this scan's frames if
        it has not been located yet) and then frozen on the fitter, so the
        scan's calibration and its samples always share it. Returns None only
        if the rows cannot be determined.
        """
        try:
            frame = measurements[0].frame_andor if measurements else None
            return self.spectrum_fitter.get_selected_rows(frame)
        except Exception as e:
            log.info(f"Could not record the sline rows for this scan: {e}")
            return None

    def _elastic_anchors_if_required(self) -> ElasticAnchors | None:
        """Anchors for fitting models that need them (na_lorentzian*); None otherwise.
        Raises if such a model is selected without a calibration (no fallback)."""
        if self.is_reference_mode:
            return None
        if not model_requires_anchors(self.spectrum_fitter.sample_config.fitting_model):
            return None
        if self.calibration_calculator is None:
            raise ValueError(
                f"Sample model '{self.spectrum_fitter.sample_config.fitting_model}' requires "
                f"elastic anchors, but no calibration is loaded."
            )
        return self.calibration_calculator.elastic_anchors()

    def _reflection_background_if_required(self, px) -> np.ndarray | None:
        """The mapped reflection background for sample fits, or None.

        Only built when the sample config uses the reflection
        background (the prmr preset). The packaged template is registered onto
        the CURRENT calibration in frequency space, so it survives VIPA
        realignment; raises if no calibration is loaded (no fallback)."""
        if self.is_reference_mode:
            return None
        if not config_requires_reflection_background(self.spectrum_fitter.sample_config):
            return None
        if self.calibration_calculator is None:
            raise ValueError(
                "Background 'reflection' requires a calibration to "
                "register the reflection background, but none is loaded."
            )
        sline_config = self.spectrum_fitter.sline_config
        n_rows = (sline_config.n_rows if sline_config.row_selection == "auto"
                  else len(sline_config.selected_rows))
        key = (id(self.calibration_calculator), n_rows)
        if self._reflection_mapper is None or self._reflection_mapper_key != key:
            if self._reflection_bg is None:
                self._reflection_bg = ReflectionBackground.load_default()
            self._reflection_mapper = ReflectionBackgroundMapper(
                self._reflection_bg, self.calibration_calculator,
                n_rows=n_rows)
            self._reflection_mapper_key = key
        return self._reflection_mapper.render(px)

    def update_calibration_calculator(self):
        if self.calibration_data is None:
            self.calibration_poly_fit_params = None
            self.calibration_calculator = None
        else:
            # Same fitter as the live frames: it holds the frozen row band, and
            # the band must not move between a calibration and its samples.
            self.calibration_poly_fit_params = calibrate(data=self.calibration_data,
                                                         poyfit_degree=self.calibration_config.degree,
                                                         fitter=self.spectrum_fitter)
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

        axial_scan = AxialScan(
            i=self._i_axial_scans,
            id=request_axial_scan.id,
            measurements=all_results,
            system_state=self.get_current_system_state(),
            calibration_params=self.calibration_poly_fit_params,
            calibration_data=self._calibration_data_to_store(),
            eye_tracker_results=request_axial_scan.eye_tracker_results,
            reflection_result_forwards=reflection_result_forwards,
            reflection_result_backwards=reflection_result_backwards,
            sline_rows=self._sline_rows_for_scan(all_results),
        )
        self.axial_scan_dict[axial_scan.i] = axial_scan

        return True


    @staticmethod
    def _accept_crossing(
        result: ReflectionResult | None,
        *,
        reference_z_um: float,
        gate_um: float,
        reference_peak: float | None,
        min_peak_fraction: float,
        reference_name: str,
    ) -> tuple[bool, str | None]:
        """
        Decide whether a sweep-scan crossing is the real surface.

        Two independent gates, both needed:
          - DISTANCE from a recent reference. Catches a crossing that is
            plausible in shape but in the wrong place.
          - PEAK AMPLITUDE relative to a reference peak. Every false crossing
            observed on 2026-07-30 was a WEAK peak while genuine ones stayed
            above 0.8x their reference: the plastic cuvette's back wall came in
            at 0.12x, and a finder outlier at 0.006x. Amplitude separates these
            by a wide margin and, unlike distance, needs no trade-off against
            how far the eye may really have moved.

        Returns (accepted, reason_if_rejected).
        """
        if result is None or not result.found:
            return False, "no crossing found"

        delta = result.event_z_um - reference_z_um
        if abs(delta) > gate_um:
            return False, (f"{delta:+.1f} µm from {reference_name} "
                           f"(gate {gate_um:.0f} µm)")

        if reference_peak and result.peak_value is not None:
            frac = result.peak_value / reference_peak
            if frac < min_peak_fraction:
                return False, (f"peak {result.peak_value:.3f} V is {frac:.2f}× the "
                               f"{reference_name} peak {reference_peak:.3f} V "
                               f"(min {min_peak_fraction:.2f}×)")

        return True, None

    def take_sweep_scan(self, request: RequestSweepScan) -> bool:
        """
        In-out sweep scan: repeated find-measure-find cycles.

        One cycle: search inward through the corneal reflection, park at the
        in-crossing + target_depth_um, snap a frame, continue inward
        approach_um past the plane, search outward (recording the
        out-crossing), park approach_um outside the freshest plane estimate,
        turn around. The two crossings of a cycle bracket the frame in time;
        depth labels ((in+out)/2 vs single-crossing) are computed in analysis,
        NOT here — both crossings are stored raw per cycle in sweep_cycles.

        Search speed/detection parameters come from the shared axial
        ScanningConfig; cycle geometry from SweepScanConfig. z_offset_um of
        the plain finder is intentionally NOT applied — the sweep scan's
        target_depth_um replaces it.
        """
        if self.is_reference_mode:
            log.info("[Sweep Scan] System is in Reference Mode - Change to Sample Mode.")
            return False

        sw = self._sweep_scan_config
        lens_x0 = self.zaber_eye_lens.get_position()

        log.info(f"[Sweep Scan] Starting: {sw.n_repeats} cycles, "
                 f"target depth {sw.target_depth_um} µm, "
                 f"approach {sw.approach_um} µm, ID: {request.id}")

        # Initial full-distance find (normal finder settings) to bootstrap.
        r0: ReflectionResult = self.find_reflection_plane(is_go_forwards=True)
        if not r0.found:
            log.info("[Sweep Scan] Initial reflection find failed - aborting.")
            self.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
            return False
        plane_est = r0.event_z_um
        # Amplitude reference for the in-crossings. The out-crossing of each
        # cycle is instead judged against that cycle's own in-crossing, so a
        # slow legitimate change in signal does not accumulate into a rejection.
        ref_peak = r0.peak_value
        log.info(f"[Sweep Scan] Plane at {plane_est:.1f} µm, reference peak "
                 f"{ref_peak:.3f} V (crossings must exceed "
                 f"{sw.min_peak_fraction * ref_peak:.3f} V).")

        measurements: list[MeasurementPoint] = []
        cycles: list[SweepCycle] = []

        for k in range(sw.n_repeats):
            if self.f2b_cancel_callback():
                log.info(f"[Sweep Scan] Cancelled during cycle {k + 1}. "
                         f"Returning lens to starting position.")
                self.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
                return False

            log.info(f"[Sweep Scan] Cycle {k + 1}/{sw.n_repeats}")

            # Park outside the current plane estimate and search inward.
            self.zaber_eye_lens.move_abs(plane_est - sw.approach_um)
            r_in: ReflectionResult = self.find_reflection_plane(is_go_forwards=True)
            in_ok, in_why = self._accept_crossing(
                r_in,
                reference_z_um=plane_est,
                gate_um=sw.plausibility_gate_um,
                reference_peak=ref_peak,
                min_peak_fraction=sw.min_peak_fraction,
                reference_name="the last plane estimate",
            )
            if r_in.found and not in_ok:
                log.warning(f"[Sweep Scan] Cycle {k + 1}: in-crossing at "
                            f"{r_in.event_z_um:.1f} µm rejected - {in_why}.")

            measurement_index = None
            r_out: ReflectionResult | None = None

            if in_ok:
                plane_est = r_in.event_z_um

                # Park at the target depth and take the frame (lens stopped).
                self.zaber_eye_lens.move_abs(plane_est + sw.target_depth_um)
                time.sleep(sw.settle_s)
                zaber_pos = self.zaber_eye_lens.get_position()
                self.b2f_emit_update_zaber_lens_position(zaber_pos)

                frame = self._get_andor_camera_snap()
                self.display_spectrum(frame=frame)
                measurements.append(
                    MeasurementPoint(
                        frame_andor=frame,
                        lens_zaber_position=zaber_pos,
                        time_stamp=time.perf_counter())
                )
                measurement_index = len(measurements) - 1

                # Continue inward past the plane, then search outward. The
                # out-crossing is judged against THIS cycle's in-crossing —
                # only ~1 s old, so both gates can be tight.
                self.zaber_eye_lens.move_abs(plane_est + sw.approach_um)
                r_out = self.find_reflection_plane(is_go_forwards=False)
                out_ok, out_why = self._accept_crossing(
                    r_out,
                    reference_z_um=r_in.event_z_um,
                    gate_um=sw.out_gate_um,
                    reference_peak=r_in.peak_value,
                    min_peak_fraction=sw.min_peak_fraction,
                    reference_name="this cycle's in-crossing",
                )
                if r_out.found and not out_ok:
                    log.warning(f"[Sweep Scan] Cycle {k + 1}: out-crossing at "
                                f"{r_out.event_z_um:.1f} µm rejected - {out_why}.")
                if out_ok:
                    # Freshest estimate for aiming the next cycle. The
                    # bias-free (in+out)/2 label is computed in analysis.
                    plane_est = r_out.event_z_um
                else:
                    log.info(f"[Sweep Scan] Cycle {k + 1}: no valid out-crossing - "
                             f"frame keeps its single-crossing (in) reference.")
            else:
                log.info(f"[Sweep Scan] Cycle {k + 1}: no valid in-crossing - "
                         f"skipping the frame this cycle.")

            cycles.append(SweepCycle(
                cycle_index=k,
                reflection_in=r_in,
                reflection_out=r_out,
                measurement_index=measurement_index,
            ))

        # Park outside the plane, then return the lens to its start position.
        self.move_and_update_gui_zaber_eye_lens_abs(lens_x0)

        n_frames = len(measurements)
        n_pairs = sum(1 for c in cycles
                      if c.measurement_index is not None
                      and c.reflection_out is not None and c.reflection_out.found)
        log.info(f"[Sweep Scan] Done: {n_frames}/{sw.n_repeats} frames taken, "
                 f"{n_pairs} with a full in/out pair.")

        self._i_axial_scans += 1
        axial_scan = AxialScan(
            i=self._i_axial_scans,
            id=request.id,
            measurements=measurements,
            system_state=self.get_current_system_state(),
            calibration_params=self.calibration_poly_fit_params,
            eye_tracker_results=request.eye_tracker_results,
            reflection_result_forwards=r0,
            reflection_result_backwards=None,
            calibration_data=self._calibration_data_to_store(),
            sweep_cycles=cycles,
            sweep_config=sw,
            scanning_config=self._axial_scan_config,
            sline_rows=self._sline_rows_for_scan(measurements),
        )
        self.axial_scan_dict[axial_scan.i] = axial_scan

        return n_frames > 0

    def update_sweep_scan_config(self, sweep_config: SweepScanConfig):
        self._sweep_scan_config = sweep_config
        log.info(f"[Sweep Scan] Config updated: {sweep_config}")

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

        The raw fitted width, as the peak lands on the detector — what you want
        while watching the live spectrum. The instrument-subtracted sample
        linewidth is an analysis output (see AnalyzedFreqShifts).
        """
        if self.calibration_calculator is None:
            return None, None
        else:
            return self.calibration_calculator.hwhm_ghz(fitting)


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
            log.info("System is in Reference (Calibration Mode) - Change to Sample Mode")
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
        min_samples_above = self._axial_scan_config.min_samples_above
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
            min_samples_above=min_samples_above,
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

