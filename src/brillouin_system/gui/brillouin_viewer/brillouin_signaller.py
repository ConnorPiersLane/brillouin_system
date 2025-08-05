import pickle
from contextlib import contextmanager
from enum import Enum

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QCoreApplication
import time

from PyQt5.QtWidgets import QApplication, QFileDialog

from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from brillouin_system.devices.cameras.flir.flir_config.flir_config import FLIRConfig
from brillouin_system.devices.zaber_microscope.led_config.led_config import LEDConfig
from brillouin_system.gui.brillouin_viewer.brillouin_backend import BrillouinBackend, SystemType
from brillouin_system.my_dataclasses.background_image import BackgroundImage
from brillouin_system.my_dataclasses.display_results import DisplayResults

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.human_interface_measurements import RequestAxialScan
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FindPeaksConfig, SlineFromFrameConfig, \
    FittingConfigs


class SystemState(Enum):
    IDLE = 0
    FREERUNNING = 1
    BUSY = 2



class BrillouinSignaller(QObject):

    # Log
    log_message = pyqtSignal(str)

    # Start / Stop Signals
    start_live_signal = pyqtSignal()
    stop_live_signal = pyqtSignal()

    # Bool Signals outwards
    background_available_state = pyqtSignal(bool)
    background_subtraction_state = pyqtSignal(bool)
    illumination_mode_state  = pyqtSignal(bool)
    reference_mode_state = pyqtSignal(bool)
    do_live_fitting_state = pyqtSignal(bool)
    gui_ready_received = pyqtSignal()

    # Signals outwards
    camera_settings_ready = pyqtSignal(dict)
    zaber_lens_position_updated = pyqtSignal(float)
    microwave_frequency_updated = pyqtSignal(float)
    background_data_ready = pyqtSignal(object)  # emits a BackgroundData instance
    frame_and_fit_ready = pyqtSignal(object)
    measurement_result_ready = pyqtSignal(object)
    camera_shutter_state_changed = pyqtSignal(bool)
    calibration_finished = pyqtSignal()
    calibration_result_ready = pyqtSignal(object)
    flir_frame_ready = pyqtSignal(np.ndarray)
    send_update_stored_axial_scans = pyqtSignal(list)



    # Signals to Frontend
    update_system_state_in_frontend = pyqtSignal(SystemState)


    def __init__(self, manager: BrillouinBackend):
        super().__init__()
        self.backend = manager
        self._running = False
        self._thread_active = False
        self._gui_ready = True
        self._camera_shutter_open = True
        self._is_cancel_operations = False

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


    @pyqtSlot()
    def on_gui_ready(self):
        self._gui_ready = True

    def update_gui(self):
        self.emit_is_illumination_continuous()
        self.emit_is_background_available()
        self.emit_camera_settings()
        self.emit_do_background_subtraction()
        self.emit_do_live_fitting_state()
        # ToDo: emit zaber positions
        if self.backend.system_type == SystemType.HUMAN_INTERFACE:
            self.update_zaber_lens_position(pos=self.backend.zaber_eye_lens.get_position())

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
            self.log_message.emit("Background subtraction disabled")
        elif self.backend.is_background_image_available():
            self.backend.start_background_subtraction()
            self.log_message.emit("Background subtraction enabled")
        else:
            self.log_message.emit("Cannot enable background subtraction: no background image available")
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
        self.log_message.emit(f"Do Live Fitting toggled to: {self.backend.do_live_fitting}")
        self.do_live_fitting_state.emit(self.backend.do_live_fitting)

    @pyqtSlot()
    def emit_do_live_fitting_state(self):
        self.do_live_fitting_state.emit(self.backend.do_live_fitting)

    @pyqtSlot()
    def toggle_illumination_mode(self):
        if self.backend.is_sample_illumination_continuous:
            self.backend.change_illumination_mode_to_pulsed()
            self.stop_live_view()
            self.log_message.emit("Switched to pulsed illumination")
        else:
            self.backend.change_illumination_mode_to_continuous()
            self.start_live_view()
            self.log_message.emit("Switched to continuous illumination")

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
            self.log_message.emit(f"Camera settings applied: {settings}")
        except Exception as e:
            self.log_message.emit(f"Failed to apply camera settings: {e}")

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
                self.log_message.emit("Camera shutter closed.")
            else:
                cam.open_shutter()
                self._camera_shutter_open = True
                self.log_message.emit("Camera shutter opened.")

            self.camera_shutter_state_changed.emit(self._camera_shutter_open)

        except Exception as e:
            self.log_message.emit(f"Failed to toggle camera shutter: {e}")

    def update_zaber_lens_position(self, pos: float):
        self.zaber_lens_position_updated.emit(pos)


    @pyqtSlot(float)
    def move_zaber_eye_lens_relative(self, step: float):
        #TODO: update this
        try:
            self.backend.zaber_eye_lens.move_rel(step)
            pos = self.backend.zaber_eye_lens.get_position()
            self.log_message.emit(f"Zaber Lens moved by {step} µm to {pos:.2f} µm")
            self.zaber_lens_position_updated.emit(pos)
        except Exception as e:
            self.log_message.emit(f"Zaber movement failed: {e}")

    @pyqtSlot(float)
    def set_microwave_frequency(self, freq: float):
        try:
            self.backend.microwave.set_frequency(freq)
            freq_real = self.backend.microwave.get_frequency()
            self.log_message.emit(f"Microwave frequency set to {freq_real:.3f} GHz")
            self.microwave_frequency_updated.emit(freq)
        except Exception as e:
            self.log_message.emit(f"Failed to set microwave frequency: {e}")


    @pyqtSlot()
    def emit_microwave_frequency(self):
        try:
            freq = self.backend.microwave.get_frequency()
            self.microwave_frequency_updated.emit(freq)
        except Exception as e:
            self.log_message.emit(f"Failed to read microwave frequency: {e}")

    @pyqtSlot()
    def acquire_background_image(self):

        # Stop live view
        self._running = False

        try:
            self.backend.take_bg_and_darknoise_images()
            self.background_available_state.emit(self.backend.is_background_image_available())
            self.log_message.emit("Background image acquired.")
        except Exception as e:
            self.log_message.emit(f"Failed to acquire background image: {e}")


        self.restart_live_view_when_ready()


    @pyqtSlot()
    def snap_and_fit(self):
        try:
            frame = self.backend.get_andor_frame()
            fitting: FittedSpectrum = self.backend.get_fitted_spectrum(frame)
            display = self.backend.get_display_results(frame, fitting)
            self.frame_and_fit_ready.emit(display)
        except Exception as e:
            self.log_message.emit(f"Error snapping frame: {e}")

    @pyqtSlot()
    def start_live_view(self):
        if not self.backend.is_sample_illumination_continuous:
            self.log_message.emit("Live view not started: illumination mode is pulsed.")
            return
        self.backend.init_shutters()

        if not self._thread_active:

            self._running = True
            QTimer.singleShot(0, self.run)

    # Start live view again
    def restart_live_view_when_ready(self):
        if not self._thread_active:
            self.start_live_view()
        else:
            QTimer.singleShot(100, self.restart_live_view_when_ready)

    @pyqtSlot()
    def stop_live_view(self):
        self._running = False
        if self.backend.system_type == SystemType.MICROSCOPE:
            self.backend.flir_cam_worker.stop_stream()
        self.update_system_state(new_state=SystemState.IDLE)

    @pyqtSlot()
    def run(self):
        if self._thread_active:
            return
        self._thread_active = True
        self._running = True
        self.update_system_state(new_state=SystemState.FREERUNNING)
        self.log_message.emit("Worker started live acquisition loop.")

        # Start the cameras
        if self.backend.system_type == SystemType.MICROSCOPE:
            self.backend.flir_cam_worker.start_stream(frame_handler=self._handle_flir_frame)


        while self._running:
            if self._gui_ready:
                self._gui_ready = False
                self.snap_and_fit()
            QCoreApplication.processEvents()
            time.sleep(0.03)

        self._thread_active = False
        self.log_message.emit("Worker stopped live acquisition loop.")


    def _update_gui(self, display_results: DisplayResults):
        self._gui_ready = False
        self.frame_and_fit_ready.emit(display_results)

    def _fitting_and_update_gui(self, frame: np.ndarray):
        fitting = self.backend.get_fitted_spectrum(frame)
        display_results = self.backend.get_display_results(frame, fitting)
        self._update_gui(display_results)

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

    @pyqtSlot(np.ndarray)
    def _handle_flir_frame(self, frame):
        self.flir_frame_ready.emit(frame)

    @pyqtSlot()
    def get_calibration_results(self):
        self.backend.update_calibration_calculator() # this recalculates the calibration
        self.calibration_result_ready.emit((self.backend.calibration_data, self.backend.calibration_calculator))

    def emit_display_result(self, display: DisplayResults):
        self.frame_and_fit_ready.emit(display)

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
    def take_axial_scan(self, request_axial_scan: RequestAxialScan):
        """
        Generates a series of ZaberPositions using the given axis, n, and step,
        then takes measurements and updates the GUI accordingly.
        """
        old_state = self.system_state
        self.stop_live_view()
        self.update_system_state(new_state=SystemState.BUSY)
        QCoreApplication.processEvents()

        try:
            self.backend.take_axial_scan(request_axial_scan)
            self.update_stored_axial_scans()
        finally:
            self.update_system_state(new_state=old_state)
            self.restart_live_view_when_ready()

    # Flir Camera
    @pyqtSlot(object)
    def flir_update_settings(self, flir_config: FLIRConfig):
        self.backend.update_flir_settings(flir_config)


    # Zaber Microscope
    @pyqtSlot(object)
    def update_microscope_leds(self, led_config: LEDConfig):
        self.backend.update_microscope_led_settings(led_config)

    @pyqtSlot()
    def close_all_shutters(self):
        self.backend.shutter_manager.close_all()



    # ----------------- Saving -----------------------
    @pyqtSlot()
    def save_all_axial_scans(self):
        scans = list(self.backend.axial_scan_dict.values())
        self._save_axial_scan_list_to_file(scans)

    @pyqtSlot(list)
    def save_multiple_axial_scans(self, indices: list[int]):
        scans = [
            scan for i, scan in self.backend.axial_scan_dict.items()
            if i in indices
        ]
        self._save_axial_scan_list_to_file(scans)

    def _save_axial_scan_list_to_file(self, scans: list):
        from PyQt5.QtWidgets import QFileDialog
        from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
            dataclass_to_hdf5_native_dict, save_dict_to_hdf5
        )

        if not scans:
            self.log_message.emit("[Save] No axial scans to save.")
            return

        base_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Axial Scans (base name)",
            filter="All Files (*)"
        )
        if not base_path:
            return

        try:
            # Save as Pickle
            pkl_path = base_path if base_path.endswith(".pkl") else base_path + ".pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(scans, f)
            self.log_message.emit(f"[✓] Pickle saved to: {pkl_path}")

            # Save as HDF5
            h5_path = base_path if base_path.endswith(".h5") else base_path + ".h5"
            native_dict = dataclass_to_hdf5_native_dict(scans)
            save_dict_to_hdf5(h5_path, native_dict)
            self.log_message.emit(f"[✓] HDF5 saved to: {h5_path}")

        except Exception as e:
            self.log_message.emit(f"[Error] Failed to save axial scans: {e}")

    @pyqtSlot(list)
    def remove_selected_axial_scans(self, indices: list[int]):
        removed = 0
        for index in indices:
            if index in self.backend.axial_scan_dict:
                del self.backend.axial_scan_dict[index]
                removed += 1
                self.log_message.emit(f"Removed axial scan {index}")
            else:
                self.log_message.emit(f"Scan {index} not found for removal")

        if removed > 0:
            self.update_stored_axial_scans()

    @pyqtSlot()
    def close(self):
        print("Stopping BrillouinWorker and closing hardware...")
        self._running = False

        def _wait_for_thread_and_shutdown():
            if self._thread_active:
                QTimer.singleShot(100, _wait_for_thread_and_shutdown)
            else:
                try:
                    self.backend.close()
                except Exception as e:
                    print(f"Error during backend shutdown: {e}")

        _wait_for_thread_and_shutdown()

