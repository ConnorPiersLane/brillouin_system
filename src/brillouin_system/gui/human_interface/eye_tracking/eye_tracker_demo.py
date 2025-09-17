import sys, time
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import QTimer

from eye_tracker_proxy import EyeTrackerProxy






def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert grayscale numpy image to QPixmap."""
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())

class EyeDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeTracker Shared Memory Demo")

        self.label_left = QLabel()
        self.label_right = QLabel()
        self.start_btn = QPushButton("Start Streaming")
        self.stop_btn = QPushButton("Stop Streaming")

        layout = QVBoxLayout()
        layout.addWidget(self.label_left)
        layout.addWidget(self.label_right)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        # proxy to EyeTracker worker
        self.proxy = EyeTrackerProxy(frame_shape=(240, 320), slots=8)  # smaller for demo
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_and_update)

        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)

    def start_stream(self):
        self.proxy.begin_stream()
        self.timer.start(30)  # ~33 Hz GUI update

    def stop_stream(self):
        self.timer.stop()
        self.proxy.end_stream()

    def poll_and_update(self):
        results = self.proxy.poll_frames()
        if not results:
            return
        latest = results[-1]
        left, right, pupil = latest["left"], latest["right"], latest["pupil"]

        # overlay a red dot for pupil
        pix_left = numpy_to_qpixmap(left)
        pix_right = numpy_to_qpixmap(right)
        for pix in (pix_left, pix_right):
            painter = QPainter(pix)
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(int(pupil[0]), int(pupil[1]), 5, 5)
            painter.end()

        self.label_left.setPixmap(pix_left.scaled(320, 240))
        self.label_right.setPixmap(pix_right.scaled(320, 240))

    def closeEvent(self, ev):
        self.stop_stream()
        self.proxy.shutdown()
        ev.accept()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    app = QApplication(sys.argv)
    demo = EyeDemo()
    demo.show()
    sys.exit(app.exec_())
