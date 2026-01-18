# dual_camera_demo.py
import sys, threading, queue
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from dual_camera_proxy import DualCameraProxy

def numpy_to_qpixmap_gray(arr: np.ndarray) -> QPixmap:
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())

class DualCamDemo(QWidget):
    def __init__(self, use_dummy=True):
        super().__init__()
        self.setWindowTitle("Dual Camera Shared-Memory Demo")

        # UI
        self.left_label = QLabel("Left")
        self.right_label = QLabel("Right")
        self.start_btn = QPushButton("Start")
        self.stop_btn  = QPushButton("Stop")

        top = QHBoxLayout()
        top.addWidget(self.left_label)
        top.addWidget(self.right_label)

        btns = QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(btns)
        self.setLayout(layout)

        # Proxy: start() also starts streaming per your implementation
        self.proxy = DualCameraProxy(dtype="uint8", slots=8, use_dummy=use_dummy)
        self.proxy.start()  # attaches SM, sends "start", waits "started"

        # Frame handoff: reader thread -> main thread
        self._frame_q = queue.Queue(maxsize=2)
        self._reader_thread = None
        self._running = False

        # GUI timer to paint latest frame (non-blocking)
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.on_tick)

        # Wire buttons
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)

    def _reader_loop(self):
        while self._running:
            try:
                left, right, ts = self.proxy.get_latest(timeout=1)  # blocking
                # keep queue small to reduce latency
                if not self._frame_q.empty():
                    try:
                        self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_q.put((left, right, ts))
            except Exception as e:
                print(f"[Demo] Reader error: {e}")
                break

    def on_start(self):
        if self._running:
            return
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self.timer.start()

    def on_stop(self):
        self.timer.stop()
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

    def on_tick(self):
        try:
            left, right, ts = self._frame_q.get_nowait()
        except queue.Empty:
            return
        self.left_label.setPixmap(numpy_to_qpixmap_gray(left))
        self.right_label.setPixmap(numpy_to_qpixmap_gray(right))

    def closeEvent(self, ev):
        try:
            self.on_stop()
        finally:
            self.proxy.shutdown()
        ev.accept()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # Windows

    app = QApplication(sys.argv)
    w = DualCamDemo(use_dummy=True)  # set False to use real hardware
    w.resize(700, 550)
    w.show()
    sys.exit(app.exec_())
