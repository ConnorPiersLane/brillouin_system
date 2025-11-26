import sys
import threading
import queue as thread_queue
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from brillouin_system.eye_tracker.eye_tracker_proxy import EyeTrackerProxy


class EyeTrackerViewer(QtWidgets.QMainWindow):
    def __init__(self, use_dummy: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Eye Tracker Viewer")
        self.resize(600, 600)

        # QLabel to show the rendered image
        self.label = QtWidgets.QLabel("Waiting for frames...", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # Proxy + threading
        self.proxy = EyeTrackerProxy(use_dummy=use_dummy)
        self.frame_queue: thread_queue.Queue[np.ndarray] = thread_queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        # Start proxy and background thread
        self._start_eye_tracker()
        self._start_frame_thread()

        # GUI timer to pull from queue and repaint
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_image_from_queue)
        self.timer.start(30)  # ~33 fps max

    # ------------------------------------------------------------------ #
    # Setup / teardown
    # ------------------------------------------------------------------ #
    def _start_eye_tracker(self):
        # This will spawn the worker process and start streaming
        self.proxy.start()

    def _start_frame_thread(self):
        def run():
            while not self._stop_event.is_set():
                try:
                    # Get latest frame triple (left, right, rendered)
                    res = self.proxy.get_latest(timeout=0.2)
                    if res is None:
                        continue
                    _, _, rendered, meta = res

                    # Non-blocking "latest-only" put to the queue
                    try:
                        # If queue is full, drop the old one
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except thread_queue.Empty:
                                pass
                        self.frame_queue.put_nowait(rendered)
                    except thread_queue.Full:
                        # Shouldn’t happen due to full() check, but just in case
                        pass
                except Exception as e:
                    print(f"[Viewer] Error in frame thread: {e}")
                    break

        self._worker_thread = threading.Thread(target=run, daemon=True)
        self._worker_thread.start()

    def closeEvent(self, event: QtGui.QCloseEvent):
        # Stop background thread
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

        # Stop worker process
        try:
            self.proxy.stop_streaming()
        except Exception:
            pass
        try:
            self.proxy.shutdown()
        except Exception:
            pass

        event.accept()

    # ------------------------------------------------------------------ #
    # GUI update
    # ------------------------------------------------------------------ #
    def _update_image_from_queue(self):
        try:
            frame = self.frame_queue.get_nowait()
        except thread_queue.Empty:
            return

        if frame is None:
            return

        # frame is assumed to be (H, W, 3) uint8, RGB
        h, w, ch = frame.shape
        assert ch == 3

        # Convert numpy array to QImage
        # QImage expects bytes in row-major order
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )

        # Optionally scale to fit the window while keeping aspect ratio
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.label.setPixmap(pixmap)


def main():
    # On some platforms (esp. Windows) with multiprocessing, you want this guard.
    app = QtWidgets.QApplication(sys.argv)

    viewer = EyeTrackerViewer(use_dummy=True)  # set False to use real cameras
    viewer.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
