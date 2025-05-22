import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QCoreApplication, QThread
import time
from brillouin_system.gui.manager.brillouin_manager import BrillouinManager
from brillouin_system.my_dataclasses.background_data import BackgroundData
from brillouin_system.my_dataclasses.calibration_data import CalibrationData

from brillouin_system.my_dataclasses.fitting_results import FittingResults
from brillouin_system.utils.calibrate_spectrometer import fit_calibration_curve


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

    # Signals outwards
    camera_settings_ready = pyqtSignal(dict)
    zaber_position_updated = pyqtSignal(float)
    microwave_frequency_updated = pyqtSignal(float)
    background_data_ready = pyqtSignal(object)  # emits a BackgroundData instance
    frame_and_fit_ready = pyqtSignal(object)
    calibration_result_ready = pyqtSignal(object)
    measurement_result_ready = pyqtSignal(list)
    camera_shutter_state_changed = pyqtSignal(bool)
    mako_frame_ready = pyqtSignal(np.ndarray)

    sd_updated = pyqtSignal(float)
    fsr_updated = pyqtSignal(float)

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
        data = BackgroundData(
            image=self.manager.bg_image,
            camera_settings=self.manager.get_camera_settings()
        )
        self.background_data_ready.emit(data)

    @pyqtSlot()
    def emit_spectral_dispersion(self):
        self.sd_updated.emit(self.manager.sd)

    @pyqtSlot()
    def emit_free_spectral_range(self):
        self.fsr_updated.emit(self.manager.fsr)

    @pyqtSlot(float)
    def apply_spectral_dispersion(self, sd: float):
        self.manager.sd = sd
        self.log_message.emit(f"Spectral dispersion updated: {sd:.6f} GHz/px")

    @pyqtSlot(float)
    def apply_free_spectral_range(self, fsr: float):
        self.manager.fsr = fsr
        self.log_message.emit(f"Free spectral range updated: {fsr:.6f} GHz")

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

        self.reference_mode_state.emit(self.manager.is_reference_mode)

    @pyqtSlot()
    def emit_camera_settings(self):
        cam = self.manager.camera
        settings = {
            "exposure": round(cam.get_exposure_time(),ndigits=4),
            "gain": cam.get_gain(),
            "roi": cam.get_roi(),
            "binning": cam.get_binning(),
        }
        self.camera_settings_ready.emit(settings)

    @pyqtSlot(dict)
    def apply_camera_settings(self, settings: dict):
        try:
            cam = self.manager.camera
            cam.set_exposure_time(settings["exposure"])
            cam.set_gain(settings["gain"])
            x0, x1, y0, y1 = settings["roi"]
            hbin, vbin = settings["binning"]
            cam.set_roi(x0, x1, y0, y1)
            cam.set_binning(hbin, vbin)

            # Reset background
            self.manager.stop_background_subtraction()
            self.manager.bg_image = None
            self.background_subtraction_state.emit(False)
            self.background_available_state.emit(False)

            self.log_message.emit(f"Camera settings applied: {settings}")
        except Exception as e:
            self.log_message.emit(f"Failed to apply camera settings: {e}")

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
            print(f"[Worker] Failed to get Zaber position for axis '{axis}': {e}")

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
        try:
            self.manager.acquire_background_image(number_of_images_to_be_taken=10)
            self.background_available_state.emit(self.manager.is_background_image_available())
            self.log_message.emit("Background image acquired.")
            self.background_data_ready.emit(BackgroundData(
                image=self.manager.bg_image,
                camera_settings=self.manager.get_camera_settings()
            ))
        except Exception as e:
            self.log_message.emit(f"Failed to acquire background image: {e}")


    @pyqtSlot()
    def snap_and_fit(self):
        try:
            fitting: FittingResults = self.manager.snap_and_get_fitting_results()
            self.frame_and_fit_ready.emit(fitting)
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

    def _update_gui(self, fitting: FittingResults):
        self._gui_ready = False
        self.frame_and_fit_ready.emit(fitting)
        # Wait for GUI to finish rendering
        start = time.time()
        while not self._gui_ready:
            QCoreApplication.processEvents()
            QThread.msleep(10)
            if time.time() - start > 5:
                self.log_message.emit("GUI timeout while rendering result.")
                break

    @pyqtSlot(list)
    def run_calibration(self, freqs: list[float]):
        calibration_fits = []

        for f in freqs:
            try:
                self.manager.microwave.set_frequency(f)
                fitting = self.manager.snap_and_get_fitting_results()
                fitting.sd = None
                fitting.fsr = None
                fitting.freq_shift_ghz = None
                calibration_fits.append(fitting)
                self._update_gui(fitting)
            except Exception as e:
                self.log_message.emit(f"[Calibration] Failed at {f:.3f} GHz: {e}")
                continue

        try:
            px_dists = [f.inter_peak_distance_px for f in calibration_fits]
            sd, fsr = fit_calibration_curve(px_dist=px_dists, freq=freqs)

            self.manager.sd = sd
            self.manager.fsr = fsr
            self.sd_updated.emit(sd)
            self.fsr_updated.emit(fsr)
            self.log_message.emit(f"[Calibration] Fit complete. SD = {sd:.5f}, FSR = {fsr:.3f}")
        except Exception as e:
            self.log_message.emit(f"[Calibration] Fit failed: {e}")
            sd, fsr = None, None

        result: CalibrationData = CalibrationData(reference_freqs_ghz=freqs, data=calibration_fits, sd=sd, fsr=fsr)

        self.calibration_result_ready.emit(result)

        return None

    @pyqtSlot(bool)
    def set_save_images_state(self, enabled: bool):
        self.manager.do_save_images = enabled
        self.log_message.emit(f"[Manager] Save Images: {enabled}")

    @pyqtSlot(int, str, float,)
    def take_measurements(self, n: int, which_axis: str, step: float):
        measurements = []
        self._gui_ready = True
        self.log_message.emit(f"Taking Measurement Series")

        for i in range(n):
            try:
                fitting = self.manager.snap_and_get_fitting_results()
                self._update_gui(fitting)

                result = self.manager.get_measurement_data(fitting)
                measurements.append(result)

                if which_axis:
                    try:
                        self.manager.zaber.move_rel(which_axis, step)
                        pos = self.manager.zaber.get_position(which_axis)
                        self.zaber_position_updated.emit(pos)
                    except Exception as e:
                        self.log_message.emit(f"Zaber move failed: {e}")

            except Exception as e:
                self.log_message.emit(f"[Measurement] Error at index {i}: {e}")

        self.measurement_result_ready.emit(measurements)

    def start_mako_stream(self):
        def on_new_frame(frame: np.ndarray):
            self.mako_frame_ready.emit(frame)
        self.manager.mako_camera.start_stream(on_new_frame)

    def stop_mako_stream(self):
        self.manager.mako_camera.stop_stream()

    @pyqtSlot()
    def close(self):
        print("Stopping BrillouinWorker and closing hardware...")
        self._running = False
        self._thread_active = False

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

        try:
            self.manager.mako_camera.close()
            print("Mako Camera closed.")
        except Exception as e:
            print(f"Error closing Mako Camera: {e}")

        print("BrillouinWorker shutdown complete.")
