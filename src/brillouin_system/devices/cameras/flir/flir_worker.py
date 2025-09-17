from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from enum import Enum
import time
import numpy as np

from brillouin_system.devices.cameras.flir.flir_config.flir_config import FLIRConfig
from brillouin_system.devices.cameras.flir.flir_base import BaseFLIRCamera


class FlirState(Enum):
    IDLE = 0
    STREAMING = 1
    SNAP_MODE = 2


class FlirWorker(QObject):
    """
    Worker that owns a FLIRCamera and runs acquisition in a background thread.
    Use start_stream(frame_handler, fps) to begin streaming.
    Use enable_software_snap() and software_snap() for manual frame grabs.
    """
    finished = pyqtSignal()

    def __init__(self, flir_camera, fps=10):
        super().__init__()
        self.cam: BaseFLIRCamera = flir_camera
        self._state = FlirState.IDLE
        self._previous_state = FlirState.IDLE
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._frame_handler = None
        self._fps = fps

        self._thread.started.connect(self._stream_loop)
        self.finished.connect(self._thread.quit)

    def start_stream(self, frame_handler: callable):
        """
        Start FLIR streaming in its own thread.

        Args:
            frame_handler: a callable that accepts a single np.ndarray frame
            fps: frames per second
        """
        if self._state == FlirState.STREAMING:
            print("[FLIRWorker] Stream already running.")
            return
        self.disable_software_snap_mode()

        self._frame_handler = frame_handler
        self._state = FlirState.STREAMING
        if not self._thread.isRunning():
            self._thread.start()

    def stop_stream(self):
        """Stop streaming and wait for thread to finish."""
        if self._state == FlirState.STREAMING:
            self._state = FlirState.IDLE

    @pyqtSlot()
    def _stream_loop(self):
        print("[FLIRWorker] Streaming loop started.")
        try:
            self.cam.start_software_stream()
            delay = 1.0 / self._fps

            while self._state == FlirState.STREAMING:
                t0 = time.time()
                try:
                    frame = self.cam.software_snap_while_stream()
                    if self._frame_handler:
                        self._frame_handler(frame)
                except Exception as e:
                    print(f"[FLIRWorker] Frame error: {e}")
                elapsed = time.time() - t0
                sleep_time = delay - elapsed

                if sleep_time < 0:
                    max_fps = 1.0 / elapsed
                    print(f"[FLIRWorker ⚠] Frame processing too slow: {elapsed:.3f}s "
                          f"(max achievable FPS ≈ {max_fps:.2f}) vs requested {self._fps} FPS")

                time.sleep(max(0, sleep_time))

        finally:
            self.cam.end_software_stream()
            self.finished.emit()
            print("[FLIRWorker] Streaming loop ended.")

    def update_fps(self, new_fps: float):
        """
        Safely update the FPS. Restarts streaming if needed.

        Args:
            new_fps: float, desired frames per second
        """
        if new_fps <= 0:
            raise ValueError("FPS must be > 0")

        self.pause_start()
        # Update fps
        self._fps = new_fps
        print(f"[FLIRWorker] FPS updated to {new_fps}")
        self.pause_end()


    def pause_start(self):
        # Save current state
        self._previous_state = self._state

        # Step 1: Pause acquisition
        if self._state == FlirState.STREAMING:
            self.stop_stream()
            self._stop_thread_safely()
        elif self._state == FlirState.SNAP_MODE:
            self.cam.end_software_stream()

        self._state = FlirState.IDLE

    def pause_end(self):
        # Resume stream if previously running
        # Step 3: Resume previous state
        if self._previous_state == FlirState.STREAMING:
            if not self._thread.isRunning():
                self._thread.start()
            self._state = FlirState.STREAMING
        elif self._previous_state == FlirState.SNAP_MODE:
            self.cam.start_software_stream()
            self._state = FlirState.SNAP_MODE
        else:
            pass


    def set_pixel_format(self, format_str):
        self.pause_start()
        self.cam.set_pixel_format(format_str=format_str)
        self.pause_end()

    def set_roi_native(self, offset_x, offset_y, width, height):
        self.pause_start()
        self.cam.set_roi_native(offset_x, offset_y, width, height)
        self.pause_end()

    def _stop_thread_safely(self):
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    def enable_software_snap(self):
        if self._state == FlirState.SNAP_MODE:
            print("[FLIRWorker] Snap mode already enabled.")
            return

        if self._state == FlirState.STREAMING:
            self.stop_stream()
            self._stop_thread_safely()

        self.cam.start_software_stream()
        self._state = FlirState.SNAP_MODE
        print("[FLIRWorker] Snap mode enabled.")

    def software_snap(self) -> np.ndarray:
        if self._state != FlirState.SNAP_MODE:
            raise RuntimeError("Software snap is not enabled. Call enable_software_snap() first.")
        try:
            return self.cam.software_snap_while_stream()
        except Exception as e:
            print(f"[FLIRWorker] Snap error: {e}")
            return None

    def disable_software_snap_mode(self):
        if self._state == FlirState.SNAP_MODE:
            self.cam.end_software_stream()
            self._state = FlirState.IDLE

    def shutdown(self):
        if self._state == FlirState.STREAMING:
            self.stop_stream()
            self._stop_thread_safely()
        elif self._state == FlirState.SNAP_MODE:
            self.cam.end_software_stream()
        self._state = FlirState.IDLE
        self.cam.shutdown()
        print("[FLIRWorker] Fully shutdown.")

    def update_exposure_gain_gamma(self, exposure_time=None, gain=None, gamma=None):
        """
        Safely update gain, exposure time, and gamma from any state.
        Restores previous state after changes.

        Args:
            gain: float, desired gain (or None to leave unchanged)
            exposure_time: float, desired exposure in microseconds (or None)
            gamma: float, desired gamma (or None)
        """
        self.pause_start()

        # Step 2: Apply settings
        try:
            if exposure_time is not None:
                self.cam.set_exposure_time(exposure_time)
            if gain is not None:
                self.cam.set_gain(gain)
            if gamma is not None:
                self.cam.set_gamma(gamma)
        except Exception as e:
            print(f"[FLIRWorker] Failed to update settings: {e}")

        #
        self.pause_end()

    def update_settings(self, flir_config: FLIRConfig):
        """
        Apply FLIRConfig settings to the camera hardware.

        Args:
            flir_config (FLIRConfig): Configuration settings to apply.
        """
        try:
            self.pause_start()
            self.cam.set_roi_native(
                offset_x=flir_config.offset_x,
                offset_y=flir_config.offset_y,
                width=flir_config.width,
                height=flir_config.height
            )
            self.cam.set_exposure_time(flir_config.exposure)
            self.cam.set_gain(flir_config.gain)
            self.cam.set_gamma(flir_config.gamma)
            self.cam.set_pixel_format(flir_config.pixel_format)

            print("[FLIRCamera] Settings successfully updated.")

        except Exception as e:
            print(f"[FLIRCamera] Failed to apply FLIRConfig: {e}")
            raise

        finally:
            self.pause_end()