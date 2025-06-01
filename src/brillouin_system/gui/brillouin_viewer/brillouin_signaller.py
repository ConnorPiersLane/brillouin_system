from contextlib import contextmanager
from typing import Callable

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QCoreApplication, QThread
import time

from brillouin_system.config.config import calibration_config
from brillouin_system.gui.brillouin_viewer.brillouin_manager import BrillouinManager
from brillouin_system.my_dataclasses.background_image import BackGroundImage

from brillouin_system.my_dataclasses.fitted_results import DisplayResults, FittedSpectrum
from brillouin_system.my_dataclasses.measurements import MeasurementSeries


@contextmanager
def force_reference_mode(manager: BrillouinManager, emit_state: Callable[[bool], None]):
    was_sample_mode = not manager.is_reference_mode
    if was_sample_mode:
        manager.change_to_reference_mode()
        emit_state(True)
        time.sleep(0.2)
    try:
        yield
    finally:
        if was_sample_mode:
            manager.change_to_sample_mode()
            emit_state(False)



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



    def __init__(self, manager: BrillouinManager):
        super().__init__()
        self.manager = manager
        self._running = False
        self._thread_active = False
        self._gui_ready = True
        self._camera_shutter_open = True


    # ------- State control -------
    # SLOTS



    @pyqtSlot()
    def on_gui_ready(self):
        self._gui_ready = True


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
        data = BackGroundImage(
            dark_image=self.manager.dark_image_sample,
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

    @pyqtSlot()
    def emit_camera_settings(self):
        cam = self.manager.camera
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
    def toggle_camera_shutter(self):
        try:
            cam = self.manager.camera

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
            self.log_message.emit(f"Microwave frequency set to {freq:.3f} GHz")
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
        #
        if self.manager.is_reference_mode:
            self.log_message.emit(f"Change to Sample Mode first to take a background image")
            return

        # Stop live view
        self._running = False

        try:
            self.manager.take_bg_image()
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
            self.snap_and_fit()
            QCoreApplication.processEvents()
            time.sleep(0.03)

        self._thread_active = False
        self.log_message.emit("Worker stopped live acquisition loop.")

    def _update_gui(self, display_results: DisplayResults):
        self._gui_ready = False
        self.frame_and_fit_ready.emit(display_results)
        # Wait for GUI to finish rendering
        start = time.time()
        while not self._gui_ready:
            QCoreApplication.processEvents()
            QThread.msleep(10)
            if time.time() - start > 5:
                self.log_message.emit("GUI timeout while rendering result.")
                break

    @pyqtSlot(bool)
    def set_save_images_state(self, do_save_images: bool):
        self.manager.do_save_images = do_save_images

    @pyqtSlot()
    def get_calibration_results(self):
        self.calibration_result_ready.emit(self.manager.calibration_results)

    def emit_display_result(self, display: DisplayResults):
        self.frame_and_fit_ready.emit(display)

    @pyqtSlot()
    def run_calibration(self):
        config = calibration_config.get()

        try:
            with force_reference_mode(self.manager, self.reference_mode_state.emit):
                success = self.manager.perform_calibration(config, on_step=self.emit_display_result)

            if success:
                self.calibration_finished.emit()
            else:
                self.log_message.emit("[Calibration] Calibration failed.")

        except Exception as e:
            self.log_message.emit(f"[Calibration] Exception: {e}")

    @pyqtSlot(int, str, float,)
    def take_measurements(self, n: int, which_axis: str, step: float):
        measurements = []
        self._gui_ready = True



        for i in range(n):
            try:
                self.log_message.emit(f"Taking Measurement {i+1}")
                frame = self.manager.get_andor_frame()
                fitting = self.manager.get_fitted_spectrum(frame)

                display_results = self.manager.get_display_results(frame, fitting)
                self._update_gui(display_results)

                result = self.manager.get_measurement_data(frame, fitting)
                measurements.append(result)

                if which_axis and not self.manager.is_reference_mode:
                    try:
                        self.manager.zaber.move_rel(which_axis, step)
                        pos = self.manager.zaber.get_position(which_axis)
                        self.zaber_position_updated.emit(pos)
                    except Exception as e:
                        self.log_message.emit(f"Zaber move failed: {e}")

            except Exception as e:
                self.log_message.emit(f"[Measurement] Error at index {i}: {e}")

        measurement_series = MeasurementSeries(measurements=measurements, calibration=self.manager.calibration_results)
        self.measurement_result_ready.emit(measurement_series)

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
                self.manager.camera.close()
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
