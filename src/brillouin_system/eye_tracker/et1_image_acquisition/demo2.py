# dual_camera_demo.py
import sys, threading, queue
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from dual_camera_proxy import DualCameraProxy
# ⬇️ import your config dialog
from brillouin_system.devices.cameras.allied.allied_config.allied_config_dialog import AlliedConfigDialog

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
        self.left_cfg_btn  = QPushButton("Left Config")
        self.right_cfg_btn = QPushButton("Right Config")

        # top row: images
        top = QHBoxLayout()
        top.addWidget(self.left_label)
        top.addWidget(self.right_label)

        # buttons
        btns = QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch()
        btns.addWidget(self.left_cfg_btn)
        btns.addWidget(self.right_cfg_btn)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(btns)
        self.setLayout(layout)

        # Proxy: start() attaches, then starts streaming
        self.proxy = DualCameraProxy(dtype="uint8", slots=8, use_dummy=use_dummy)
        self.proxy.start()

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
        self.left_cfg_btn.clicked.connect(self.open_left_config)
        self.right_cfg_btn.clicked.connect(self.open_right_config)

    # -------- Reader thread & painting --------
    def _reader_loop(self):
        while self._running:
            try:
                left, right, ts = self.proxy.get_frames()  # blocking
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
        # show true pixel size; no scaling
        self.left_label.setPixmap(numpy_to_qpixmap_gray(left))
        self.right_label.setPixmap(numpy_to_qpixmap_gray(right))

    # -------- Config dialogs --------
    def open_left_config(self):
        # The dialog will load from allied_config["left"] internally and call our callback on Apply
        dlg = AlliedConfigDialog("left", self._apply_left_config, parent=self)
        dlg.exec_()

    def open_right_config(self):
        dlg = AlliedConfigDialog("right", self._apply_right_config, parent=self)
        dlg.exec_()

    def _apply_left_config(self, cfg_obj):
        """
        Called by AlliedConfigDialog on Apply for the left side.
        cfg_obj is an AlliedConfig (the dialog returns allied_config['left'].get()).
        We pause the reader to avoid event contention, push the update, then resume.
        """
        was_running = self._running
        if was_running:
            self.on_stop()
        try:
            # Only left changes; right stays as-is (None)
            self.proxy.set_configs(cfg_left=cfg_obj, cfg_right=None)
            print("[Demo] Left config applied.")
        finally:
            if was_running:
                self.on_start()

    def _apply_right_config(self, cfg_obj):
        was_running = self._running
        if was_running:
            self.on_stop()
        try:
            self.proxy.set_configs(cfg_left=None, cfg_right=cfg_obj)
            print("[Demo] Right config applied.")
        finally:
            if was_running:
                self.on_start()

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
    w.resize(900, 500)
    w.show()
    sys.exit(app.exec_())
