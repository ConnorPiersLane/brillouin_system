from contextlib import contextmanager

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QCoreApplication
import time

from PyQt5.QtWidgets import QApplication

from brillouin_system.config.config import calibration_config
from brillouin_system.gui.brillouin_viewer.brillouin_backend import BrillouinBackend
from brillouin_system.my_dataclasses.background_image import BackgroundImage

from brillouin_system.my_dataclasses.fitted_results import DisplayResults, FittedSpectrum
from brillouin_system.my_dataclasses.measurements import MeasurementSettings, MeasurementSeries
from brillouin_system.my_dataclasses.zaber_position import generate_zaber_positions


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
    zaber_position_updated = pyqtSignal(float)
    microwave_frequency_updated = pyqtSignal(float)
    background_data_ready = pyqtSignal(object)  # emits a BackgroundData instance
    frame_and_fit_ready = pyqtSignal(object)
    measurement_result_ready = pyqtSignal(object)
    camera_shutter_state_changed = pyqtSignal(bool)
    calibration_finished = pyqtSignal()
    calibration_result_ready = pyqtSignal(object)

    # Flir Camera Signals, from Backend to Frontend


    def __init__(self, manager: BrillouinBackend):
        super().__init__()
        self.manager = manager
        self._running = False
        self._thread_active = False
        self._gui_ready = True
        self._camera_shutter_open = True
        self._is_cancel_operations = False



    # ------- State control -------
    # SLOTS

    @contextmanager
    def force_reference_mode(self):
        was_sample_mode = not self.manager.is_reference_mode
        if was_sample_mode:
            self.manager.change_to_reference_mode()
            self.reference_mode_state.emit(True)
            time.sleep(0.2)
        try:
            yield
        finally:
            if was_sample_mode:
                self.manager.change_to_sample_mode()
                self.reference_mode_state.emit(False)

    @pyqtSlot()
    def on_gui_ready(self):
        self._gui_ready = True

    def update_gui(self):
        self.emit_is_illumination_continuous()
        self.emit_is_background_available()
        self.emit_camera_settings()
        self.emit_do_background_subtraction()
        self.emit_do_live_fitting_state()

    @pyqtSlot()
    def emit_do_background_subtraction(self):
        self.background_subtraction_state.emit(self.manager.do_background_subtraction)

    @pyqtSlot()
    def emit_is_background_available(self):
        self.background_available_state.emit(self.manager.is_background_image_available())

    @pyqtSlot()
    def emit_is_illumination_continuous(self):
        self.illumination_mode_state.emit(self.manager.is_sample_illumination_continuous)

    # Toggle
    @pyqtSlot()
    def toggle_background_subtraction(self):
        if self.manager.do_background_subtraction:
            self.manager.stop_background_subtraction()
            self.log_message.emit("Background subtraction disabled")
        elif self.manager.is_background_image_available():
            self.manager.start_background_subtraction()
            self.log_message.emit("Background subtraction enabled")
        else:
            self.log_message.emit("Cannot enable background subtraction: no background image available")
            return

        # Emit updated state to viewer
        self.background_subtraction_state.emit(self.manager.do_background_subtraction)
        self.background_available_state.emit(self.manager.is_background_image_available())


    @pyqtSlot()
    def emit_background_data(self):
        data = BackgroundImage(
            dark_image=self.manager.dark_image,
            bg_image=self.manager.bg_image,
        )
        self.background_data_ready.emit(data)

    @pyqtSlot()
    def toggle_do_live_fitting(self):
        self.manager.do_live_fitting = not self.manager.do_live_fitting
        self.log_message.emit(f"Do Live Fitting toggled to: {self.manager.do_live_fitting}")
        self.do_live_fitting_state.emit(self.manager.do_live_fitting)

    @pyqtSlot()
    def emit_do_live_fitting_state(self):
        self.do_live_fitting_state.emit(self.manager.do_live_fitting)

    @pyqtSlot()
    def toggle_illumination_mode(self):
        if self.manager.is_sample_illumination_continuous:
            self.manager.change_illumination_mode_to_pulsed()
            self.stop_live_view()
            self.log_message.emit("Switched to pulsed illumination")
        else:
            self.manager.change_illumination_mode_to_continuous()
            self.start_live_view()
            self.log_message.emit("Switched to continuous illumination")

        self.illumination_mode_state.emit(self.manager.is_sample_illumination_continuous)

    @pyqtSlot()
    def toggle_reference_mode(self):
        if self.manager.is_reference_mode:
            self.manager.change_to_sample_mode()
        else:
            self.manager.change_to_reference_mode()
        self.emit_camera_settings()
        self.reference_mode_state.emit(self.manager.is_reference_mode)
        # Emit updated state to viewer
        self.background_subtraction_state.emit(self.manager.do_background_subtraction)
        self.background_available_state.emit(self.manager.is_background_image_available())

    @pyqtSlot()
    def emit_camera_settings(self):
        cam = self.manager.andor_camera
        settings = {
            "exposure": round(cam.get_exposure_time(),ndigits=4),
            "gain": cam.get_emccd_gain(),
        }
        self.camera_settings_ready.emit(settings)

    @pyqtSlot(dict)
    def apply_camera_settings(self, settings: dict):
        try:
            self.manager.set_camera_settings(
                exposure_time=settings["exposure"],
                emccd_gain=settings["gain"],
            )
            self.log_message.emit(f"Camera settings applied: {settings}")
        except Exception as e:
            self.log_message.emit(f"Failed to apply camera settings: {e}")

        # Remove Background if sample mode
        if not self.manager.is_reference_mode:
            # Reset background
            self.manager.stop_background_subtraction()
            self.manager.bg_image = None
            self.background_subtraction_state.emit(False)
            self.background_available_state.emit(False)


    @pyqtSlot()
    def reload_andor_config(self):
        self.manager.reload_andor_config()

    @pyqtSlot()
    def toggle_camera_shutter(self):
        try:
            cam = self.manager.andor_camera

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

    @pyqtSlot(str)
    def query_zaber_position(self, axis: str):
        try:
            pos = self.manager.zaber.get_position(axis)
            self.zaber_position_updated.emit(pos)
        except Exception as e:
            self.log_message.emit(f"Failed to get Zaber position for axis '{axis}': {e}")

    @pyqtSlot(str, float)
    def move_zaber_relative(self, which_axis: str, step: float):
        try:
            self.manager.zaber.move_rel(which_axis, step)
            pos = self.manager.zaber.get_position(which_axis)
            self.log_message.emit(f"Zaber {which_axis}-Axis moved by {step} µm to {pos:.2f} µm")
            self.zaber_position_updated.emit(pos)
        except Exception as e:
            self.log_message.emit(f"Zaber movement failed: {e}")

    @pyqtSlot(float)
    def set_microwave_frequency(self, freq: float):
        try:
            self.manager.microwave.set_frequency(freq)
            freq_real = self.manager.microwave.get_frequency()
            self.log_message.emit(f"Microwave frequency set to {freq_real:.3f} GHz")
            self.microwave_frequency_updated.emit(freq)
        except Exception as e:
            self.log_message.emit(f"Failed to set microwave frequency: {e}")


    @pyqtSlot()
    def emit_microwave_frequency(self):
        try:
            freq = self.manager.microwave.get_frequency()
            self.microwave_frequency_updated.emit(freq)
        except Exception as e:
            self.log_message.emit(f"Failed to read microwave frequency: {e}")

    @pyqtSlot()
    def acquire_background_image(self):

        # Stop live view
        self._running = False

        try:
            self.manager.take_bg_and_darknoise_images()
            self.background_available_state.emit(self.manager.is_background_image_available())
            self.log_message.emit("Background image acquired.")
        except Exception as e:
            self.log_message.emit(f"Failed to acquire background image: {e}")

        # Start live view again
        def restart_live_view_when_ready():
            if not self._thread_active:
                self.start_live_view()
            else:
                QTimer.singleShot(100, restart_live_view_when_ready)
        restart_live_view_when_ready()


    @pyqtSlot()
    def snap_and_fit(self):
        try:
            frame = self.manager.get_andor_frame()
            fitting: FittedSpectrum = self.manager.get_fitted_spectrum(frame)
            display = self.manager.get_display_results(frame, fitting)
            self.frame_and_fit_ready.emit(display)
        except Exception as e:
            self.log_message.emit(f"Error snapping frame: {e}")

    @pyqtSlot()
    def start_live_view(self):
        if not self.manager.is_sample_illumination_continuous:
            self.log_message.emit("Live view not started: illumination mode is pulsed.")
            return

        if not self._thread_active:
            self._running = True
            QTimer.singleShot(0, self.run)

    @pyqtSlot()
    def stop_live_view(self):
        self._running = False

    @pyqtSlot()
    def run(self):
        if self._thread_active:
            return
        self._thread_active = True
        self._running = True
        self.log_message.emit("Worker started live acquisition loop.")

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
        fitting = self.manager.get_fitted_spectrum(frame)
        display_results = self.manager.get_display_results(frame, fitting)
        self._update_gui(display_results)

    @pyqtSlot()
    def cancel_operations(self):
        self._is_cancel_operations = True

    @pyqtSlot()
    def reset_cancel(self):
        self._is_cancel_operations = False
        self.update_gui()

    @pyqtSlot()
    def get_calibration_results(self):
        self.manager.update_calibration_calculator() # this recalculates the calibration
        self.calibration_result_ready.emit((self.manager.calibration_data, self.manager.calibration_calculator))

    def emit_display_result(self, display: DisplayResults):
        self.frame_and_fit_ready.emit(display)

    @pyqtSlot()
    def run_calibration(self):
        config = calibration_config.get()

        try:
            with self.force_reference_mode():
                success = self.manager.perform_calibration(config, call_update_gui=self.emit_display_result)

            if success:
                self.calibration_finished.emit()
            else:
                self.log_message.emit("[Calibration] Calibration failed.")

        except Exception as e:
            self.log_message.emit(f"[Calibration] Exception: {e}")

    @pyqtSlot(object)
    def take_measurements(self, measurement_settings: MeasurementSettings):
        """
        Generates a series of ZaberPositions using the given axis, n, and step,
        then takes measurements and updates the GUI accordingly.
        """
        try:
            self._gui_ready = True
            measurements = []

            # Extract axis and movement parameters
            which_axis = measurement_settings.move_axes
            start = self.manager.zaber.get_position(which_axis)
            step = measurement_settings.move_x_rel_um
            n = measurement_settings.n_measurements
            fixed_positions = {}

            # Generate the list of positions to move to
            zaber_positions = generate_zaber_positions(
                axis=which_axis,
                start=start,
                step=step,
                n=n,
                fixed_positions=fixed_positions
            )

            for i, zaber_pos in enumerate(zaber_positions):
                QApplication.processEvents()
                # Check if user requested cancelation
                if self._is_cancel_operations:
                    self.reset_cancel()
                    raise InterruptedError(f"Cancelled at measurement {i}")

                try:
                    self.log_message.emit(
                        f"Measurement {i + 1}: "
                        f"Zaber (x,y,z) = ({zaber_pos.x:.2f},{zaber_pos.y:.2f},{zaber_pos.z:.2f}) µm"
                    )
                    # Take single measurement
                    mp = self.manager.take_one_measurement(zaber_position=zaber_pos)
                    self._fitting_and_update_gui(frame=mp.frame)
                    measurements.append(mp)

                except Exception as e:
                    self.log_message.emit(f"[Measurement] Error at index {i}: {e}")

            series = MeasurementSeries(
                measurements=measurements,
                state_mode=self.manager.get_current_state_mode(),
                calibration_data=self.manager.calibration_data,
                settings=measurement_settings,
            )

            # Emit full result list if all measurements completed
            self.measurement_result_ready.emit(series)

        except InterruptedError as e:
            self.log_message.emit(f"[Measurement] Cancelled by user: {e}")
        except Exception as e:
            self.log_message.emit(f"[Measurement] Exception: {e}")


    # Flir Camera
    @pyqtSlot()
    def update_flir_exposure_gain_gamma(self, exposure_time, gain, gamma):
        pass




    @pyqtSlot()
    def close(self):
        print("Stopping BrillouinWorker and closing hardware...")
        self._running = False

        def _wait_for_thread_and_shutdown():
            if self._thread_active:
                QTimer.singleShot(100, _wait_for_thread_and_shutdown)
            else:
                _shutdown_devices()

        def _shutdown_devices():
            try:
                self.manager.andor_camera.close()
                print("Camera closed.")
            except Exception as e:
                print(f"Error closing camera: {e}")

            try:
                self.manager.shutter_manager.close_all()
                print("Shutter manager closed.")
            except Exception as e:
                print(f"Error closing shutter manager: {e}")

            try:
                self.manager.microwave.shutdown()
                print("Microwave closed.")
            except Exception as e:
                print(f"Error closing microwave: {e}")

            try:
                self.manager.zaber.close()
                print("Zaber closed.")
            except Exception as e:
                print(f"Error closing zaber: {e}")

            print("BrillouinWorker shutdown complete.")

        _wait_for_thread_and_shutdown()
