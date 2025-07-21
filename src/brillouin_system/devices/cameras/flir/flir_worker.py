from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from enum import Enum
import time
import numpy as np


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

    def __init__(self, flir_camera):
        super().__init__()
        self.cam = flir_camera
        self._state = FlirState.IDLE
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._frame_handler = None
        self._fps = 10

        self._thread.started.connect(self._stream_loop)
        self.finished.connect(self._thread.quit)

    def start_stream(self, frame_handler: callable, fps: int = 10):
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
        self._fps = fps
        self._state = FlirState.STREAMING
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
                time.sleep(max(0, delay - (time.time() - t0)))
        finally:
            self.cam.end_software_stream()
            self.finished.emit()
            print("[FLIRWorker] Streaming loop ended.")

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
