from enum import Enum
import threading

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QCoreApplication
from PyQt5 import QtCore

from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from brillouin_system.guis.human_interface.hi_backend import HiBackend
from brillouin_system.hi_axial_scanning.hi_axial_scanning_config.axial_scanning_config import AxialScanningConfig
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.my_dataclasses.background_image import BackgroundImage
from brillouin_system.my_dataclasses.display_results import DisplayResults

from brillouin_system.my_dataclasses.human_interface_measurements import RequestAxialStepScan, RequestAxialContScan
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FittingConfigs

log = get_logger(__name__)


class SystemState(Enum):
    IDLE = 0
    FREERUNNING = 1
    BUSY = 2



class HiSignaller(QObject):


    # Start / Stop Signals
    start_live_signal = pyqtSignal()
    stop_live_signal = pyqtSignal()

    # Bool Signals outwards
    background_available_state = pyqtSignal(bool)
    background_subtraction_state = pyqtSignal(bool)
    illumination_mode_state  = pyqtSignal(bool)
    reference_mode_state = pyqtSignal(bool)
    do_live_fitting_state = pyqtSignal(bool)

    # Signals outwards
    camera_settings_ready = pyqtSignal(dict)
    zaber_lens_position_updated = pyqtSignal(float)
    microwave_frequency_updated = pyqtSignal(float)
    background_data_ready = pyqtSignal(object)  # emits a BackgroundData instance
    # frame_and_fit_ready = pyqtSignal(object)
    measurement_result_ready = pyqtSignal(object)
    camera_shutter_state_changed = pyqtSignal(bool)
    calibration_finished = pyqtSignal()
    calibration_result_ready = pyqtSignal(object)
    send_update_stored_axial_scans = pyqtSignal(list)
    axial_scan_data_ready = pyqtSignal(object)
    send_axial_scans_to_save = pyqtSignal(list)
    send_message_to_frontend = pyqtSignal(str, str)
    new_andor_display_ready = pyqtSignal()
    close_event_finished = pyqtSignal()

    zaber_stage_positions_updated = pyqtSignal(float, float, float)
    # (x, y, z) positions in µm





    # Signals to Frontend
    update_system_state_in_frontend = pyqtSignal(SystemState)

    # Eye Tracking
    eye_tracking_result_ready = pyqtSignal(object)


    def __init__(self, manager: HiBackend):
        super().__init__()
        self.backend = manager
        self._running = False
        self._thread_active = False
        self._camera_shutter_open = True
        self._is_cancel_operations = False

        self._mb_lock = threading.Lock()
        self._mb_latest_display = None

        # in HiSignaller.__init__ (after moveToThread happens in the host)
        self._acq_timer = None
        self._acq_interval_ms = 20


        self.system_state = SystemState.IDLE

        self.backend.init_f2b_signals(cancel_callback=self.is_cancel_requested)
        self.backend.init_b2f_emit_display_result(emit_display_result=self.emit_display_result)


        # Init Zaber position updates
        self.backend.init_b2f_zaber_position_updates_human_interface(
            emit_update_zaber_lens_position=self.update_zaber_lens_position
        )


    def update_system_state(self, new_state: SystemState):
        self.system_state = new_state
        self.update_system_state_in_frontend.emit(new_state)
        if new_state in (SystemState.BUSY, SystemState.FREERUNNING):
            self.reset_cancel()

    def update_gui(self):
        self.emit_is_illumination_continuous()
        self.emit_is_background_available()
        self.emit_camera_settings()
        self.emit_do_background_subtraction()
        self.emit_do_live_fitting_state()
        self.update_stage_positions() #xyz stage
        self.update_zaber_lens_position(self.backend.zaber_eye_lens.get_position()) # lens


    def _mailbox_push_andor_display(self, display):
        need_notify = False
        with self._mb_lock:
            # if GUI hasn't consumed yet, mailbox is not None: just overwrite, don't notify
            if self._mb_latest_display is None:
                need_notify = True
            self._mb_latest_display = display
        if need_notify:
            self.new_andor_display_ready.emit()  # queued across threads

    def fetch_latest_andor_display(self):
        # consumer (GUI thread): swap out and return
        with self._mb_lock:
            item = self._mb_latest_display
            self._mb_latest_display = None
        return item

    @pyqtSlot()
    def emit_do_background_subtraction(self):
        self.background_subtraction_state.emit(self.backend.do_background_subtraction)

    @pyqtSlot()
    def emit_is_background_available(self):
        self.background_available_state.emit(self.backend.is_background_image_available())

    @pyqtSlot()
    def emit_is_illumination_continuous(self):
        self.illumination_mode_state.emit(self.backend.is_sample_illumination_continuous)


    # Toggle
    @pyqtSlot()
    def toggle_background_subtraction(self):
        if self.backend.do_background_subtraction:
            self.backend.stop_background_subtraction()
            log.info("Background subtraction disabled")
        elif self.backend.is_background_image_available():
            self.backend.start_background_subtraction()
            log.info("Background subtraction enabled")
        else:
            log.info("Cannot enable background subtraction: no background image available")
            return

        # Emit updated state to viewer
        self.background_subtraction_state.emit(self.backend.do_background_subtraction)
        self.background_available_state.emit(self.backend.is_background_image_available())


    @pyqtSlot()
    def emit_background_data(self):
        data = BackgroundImage(
            dark_image=self.backend.dark_image,
            bg_image=self.backend.bg_image,
        )
        self.background_data_ready.emit(data)

    @pyqtSlot()
    def toggle_do_live_fitting(self):
        self.backend.do_live_fitting = not self.backend.do_live_fitting
        log.info(f"Do Live Fitting toggled to: {self.backend.do_live_fitting}")
        self.do_live_fitting_state.emit(self.backend.do_live_fitting)

    @pyqtSlot()
    def emit_do_live_fitting_state(self):
        self.do_live_fitting_state.emit(self.backend.do_live_fitting)

    @pyqtSlot()
    def toggle_illumination_mode(self):
        if self.backend.is_sample_illumination_continuous:
            self.backend.change_illumination_mode_to_pulsed()
            self.stop_live_view()
            log.info("Switched to pulsed illumination")
        else:
            self.backend.change_illumination_mode_to_continuous()
            self.start_live_view()
            log.info("Switched to continuous illumination")

        self.illumination_mode_state.emit(self.backend.is_sample_illumination_continuous)

    @pyqtSlot()
    def toggle_reference_mode(self):
        if self.backend.is_reference_mode:
            self.backend.change_to_sample_mode()
        else:
            self.backend.change_to_reference_mode()
        self.emit_camera_settings()
        self.reference_mode_state.emit(self.backend.is_reference_mode)
        # Emit updated state to viewer
        self.background_subtraction_state.emit(self.backend.do_background_subtraction)
        self.background_available_state.emit(self.backend.is_background_image_available())

    @pyqtSlot()
    def emit_camera_settings(self):
        cam = self.backend.andor_camera
        settings = {
            "exposure": round(cam.get_exposure_time(),ndigits=4),
            "gain": cam.get_emccd_gain(),
        }
        self.camera_settings_ready.emit(settings)

    @pyqtSlot(dict)
    def apply_camera_settings(self, settings: dict):
        try:
            self.backend.set_andor_exposure(
                exposure_time=settings["exposure"],
                emccd_gain=settings["gain"],
            )
            log.info(f"Camera settings applied: {settings}")
        except Exception as e:
            log.info(f"Failed to apply camera settings: {e}")

        # Remove Background if sample mode
        if not self.backend.is_reference_mode:
            # Reset background
            self.backend.stop_background_subtraction()
            self.backend.bg_image = None
            self.background_subtraction_state.emit(False)
            self.background_available_state.emit(False)


    @pyqtSlot(object)
    def update_andor_config_settings(self, andor_config: AndorConfig):
        self.backend.update_andor_config_settings(andor_config)

    @pyqtSlot(object)
    def update_axial_scan_settings(self, axial_scan_settings: AxialScanningConfig):
        self.backend.update_axial_scan_settings(axial_scan_settings)

    @pyqtSlot(FittingConfigs)
    def update_fitting_configs(self, fitting_configs: FittingConfigs):
        self.backend.spectrum_fitter.update_configs(fitting_configs)

    @pyqtSlot()
    def toggle_camera_shutter(self):
        try:
            cam = self.backend.andor_camera

            if self._camera_shutter_open:
                cam.close_shutter()
                self._camera_shutter_open = False
                log.info("Camera shutter closed.")
            else:
                cam.open_shutter()
                self._camera_shutter_open = True
                log.info("Camera shutter opened.")

            self.camera_shutter_state_changed.emit(self._camera_shutter_open)

        except Exception as e:
            log.warning(f"Failed to toggle camera shutter: {e}")

    def update_zaber_lens_position(self, pos: float):
        self.zaber_lens_position_updated.emit(pos)

    @pyqtSlot(float)
    def move_zaber_stage_x_relative(self, step: float):
        def do():
            self.backend.zaber_hi.move_rel(dx=step)
            self.update_stage_positions()

        self._pause_acq_for(do)

    @pyqtSlot(float)
    def move_zaber_stage_y_relative(self, step: float):
        self._pause_acq_for(lambda: (self.backend.zaber_hi.move_rel(dy=step),
                                     self.update_stage_positions()))

    @pyqtSlot(float)
    def move_zaber_stage_z_relative(self, step: float):
        self._pause_acq_for(lambda: (self.backend.zaber_hi.move_rel(dz=step),
                                     self.update_stage_positions()))

    @pyqtSlot(float)
    def move_zaber_eye_lens_relative(self, step: float):
        def do():
            self.backend.zaber_eye_lens.move_rel(step)
            pos = self.backend.zaber_eye_lens.get_position()
            self.zaber_lens_position_updated.emit(pos)

        self._pause_acq_for(do)

    def update_stage_positions(self):
        try:
            x, y, z = self.backend.zaber_hi.get_position()
            self.zaber_stage_positions_updated.emit(x, y, z)
        except Exception as e:
            log.warning(f"Failed to read stage positions: {e}")

    @pyqtSlot(float)
    def set_microwave_frequency(self, freq: float):
        try:
            self.backend.microwave.set_frequency(freq)
            freq_real = self.backend.microwave.get_frequency()
            log.info(f"Microwave frequency set to {freq_real:.3f} GHz")
            self.microwave_frequency_updated.emit(freq)
        except Exception as e:
            log.warning(f"Failed to set microwave frequency: {e}")


    @pyqtSlot()
    def emit_microwave_frequency(self):
        try:
            freq = self.backend.microwave.get_frequency()
            self.microwave_frequency_updated.emit(freq)
        except Exception as e:
            log.warning(f"Failed to read microwave frequency: {e}")

    @pyqtSlot()
    def acquire_background_image(self):

        # Stop live view
        self._running = False

        try:
            self.backend.take_bg_and_darknoise_images()
            self.background_available_state.emit(self.backend.is_background_image_available())
            log.info("Background image acquired.")
        except Exception as e:
            log.warning(f"Failed to acquire background image: {e}")


        self.restart_live_view_when_ready()

    @pyqtSlot()
    def snap_and_fit(self):
        try:
            frame, _ = self.backend.get_andor_frame()
            self.backend.display_spectrum(frame=frame)

        except Exception as e:
            log.warning(f"Error snapping frame: {e}")

    @pyqtSlot()
    def start_live_view(self):
        if self._running:
            return
        self.backend.init_shutters()
        self._running = True
        self.update_system_state(new_state=SystemState.FREERUNNING)

        if self._acq_timer is None:
            self._acq_timer = QtCore.QTimer(self)  # parent = self (lives in worker)
            self._acq_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
            self._acq_timer.timeout.connect(self._acq_tick)  # local, same-thread

        self._acq_timer.start(self._acq_interval_ms)

    @pyqtSlot()
    def stop_live_view(self):
        self._running = False
        if self._acq_timer:
            self._acq_timer.stop()
        self.update_system_state(new_state=SystemState.IDLE)

    def restart_live_view_when_ready(self):
        QTimer.singleShot(0, self.start_live_view)


    def _pause_acq_for(self, fn):
        was_running = self._running
        if self._acq_timer:
            self._acq_timer.stop()
        self._running = False
        try:
            return fn()
        finally:
            if was_running:
                self._running = True
                if self._acq_timer:
                    self._acq_timer.start(self._acq_interval_ms)

    def _acq_tick(self):
        # run at most one acq at a time; coalesce timeouts
        if not self._running or self._acq_timer is None:
            return

        self._acq_timer.stop()

        try:
            with self._mb_lock:
                if self._mb_latest_display is not None:
                    return  # GUI hasn't consumed yet; skip doing work
            self.snap_and_fit()
        finally:
            # schedule the next tick after we’re done (coalesced)
            if self._running and self._acq_timer:
                self._acq_timer.start(self._acq_interval_ms)

    def cancel_operations(self):
        self._is_cancel_operations = True

    def update_stored_axial_scans(self):
        lines = self.backend.get_list_of_axial_scans()
        self.send_update_stored_axial_scans.emit(lines)

    # ------- State control -------
    # SLOTS
    # in BrillouinSignaller
    #  cancel callback to pass down to the backend
    def is_cancel_requested(self) -> bool:
        return self._is_cancel_operations


    @pyqtSlot()
    def reset_cancel(self):
        self._is_cancel_operations = False


    @pyqtSlot()
    def get_calibration_results(self):
        self.backend.update_calibration_calculator() # this recalculates the calibration
        self.calibration_result_ready.emit((self.backend.calibration_data, self.backend.calibration_calculator))

    def emit_display_result(self, display: DisplayResults):
        self._mailbox_push_andor_display(display)


    @pyqtSlot()
    def run_calibration(self):
        old_state = self.system_state
        self.update_system_state(new_state=SystemState.BUSY)
        try:
            is_sucess = self.backend.perform_calibration()
            if is_sucess:
                self.calibration_finished.emit()
        finally:
            self.update_system_state(new_state=old_state)

    @pyqtSlot(object)
    def take_axial_step_scan(self, request_axial_scan: RequestAxialStepScan):
        """
        Generates a series of ZaberPositions using the given axis, n, and step,
        then takes measurements and updates the GUI accordingly.
        """
        if self.backend.calibration_poly_fit_params is None:
            self.send_message_to_user('Warning', "No Calibration available. Run Calibration first.")
            return

        old_state = self.system_state
        self.stop_live_view()
        self.update_system_state(new_state=SystemState.BUSY)
        QCoreApplication.processEvents()

        try:
            self.backend.take_axial_step_scan(request_axial_scan)
            self.update_stored_axial_scans()
        finally:
            self.update_system_state(new_state=old_state)
            self.restart_live_view_when_ready()

    @pyqtSlot(object)
    def take_axial_cont_scan(self, request_axial_scan: RequestAxialContScan):
        """
        Generates a series of ZaberPositions using the given axis, n, and step,
        then takes measurements and updates the GUI accordingly.
        """
        if self.backend.calibration_poly_fit_params is None:
            self.send_message_to_user('Warning', "No Calibration available. Run Calibration first.")
            return
        if self.backend.is_reference_mode:
            self.send_message_to_user('Warning', "System in Calibration Mode. Switch to Measurement mode first")
            return

        old_state = self.system_state
        self.stop_live_view()
        self.update_system_state(new_state=SystemState.BUSY)
        QCoreApplication.processEvents()

        try:
            self.backend.take_axial_cont_scan(request_axial_scan)
            self.update_stored_axial_scans()
        finally:
            self.update_system_state(new_state=old_state)
            self.restart_live_view_when_ready()

    @pyqtSlot(int)
    def handle_request_axial_scan_data(self, index: int):
        scan_data = self.backend.get_axial_scan_data(index)  # pure Python call
        if scan_data is not None:
            self.axial_scan_data_ready.emit(scan_data)
        else:
            log.warning(f"Requested scan index {index} not found.")

    @pyqtSlot()
    def delegate_take_and_store_bg_value_for_reflection_finding(self):
        self.backend.take_and_store_bg_value_for_reflection_finding()

    @pyqtSlot()
    def delegate_find_reflection_plane(self):
        self.backend.find_reflection_plane()


    @pyqtSlot()
    def close_all_shutters(self):
        self.backend.shutter_manager.close_all()



    # ----------------- Saving -----------------------
    @pyqtSlot()
    def save_all_axial_scans(self):
        scans = list(self.backend.axial_scan_dict.values())
        self.send_axial_scans_to_save.emit(scans)

    @pyqtSlot(list)
    def save_multiple_axial_scans(self, indices: list[int]):
        scans = [
            scan for i, scan in self.backend.axial_scan_dict.items()
            if i in indices
        ]
        self.send_axial_scans_to_save.emit(scans)


    @pyqtSlot(list)
    def remove_selected_axial_scans(self, indices: list[int]):
        removed = 0
        for index in indices:
            if index in self.backend.axial_scan_dict:
                del self.backend.axial_scan_dict[index]
                removed += 1
                log.info(f"Removed axial scan {index}")
            else:
                log.warning(f"Scan {index} not found for removal")

        if removed > 0:
            self.update_stored_axial_scans()

    @pyqtSlot()
    def send_message_to_user(self, title: str, message: str):
        self.send_message_to_frontend.emit(title, message)



    def close(self):
        print("Stopping BrillouinWorker and closing hardware...")
        self._running = False
        try:
            self.backend.close()
        except Exception as e:
            print(f"Error during backend shutdown: {e}")
        finally:
            self.close_event_finished.emit()

