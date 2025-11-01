# eye_tracker_gui.py
# Shows 4 live images from your EyeTracker with Start/Stop buttons.
# Works with PySide6 or PyQt5.
#
# HOW TO USE:
# 1) Implement `build_eye_tracker()` below so it returns a ready EyeTracker.
#    (Create/Load your EyeTrackerConfig there.)
# 2) Run: python eye_tracker_gui.py

import sys
import numpy as np

# --- Qt import that works with either PySide6 or PyQt5 ---
qt_backend = None

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QGridLayout, QHBoxLayout, QVBoxLayout, QMessageBox
)
qt_backend = "PyQt5"

# --- Import your EyeTracker class ---
# Make sure eye_tracker.py is on your PYTHONPATH or in the same folder.
from brillouin_system.eye_tracker.eye_tracker import EyeTracker, EyeTrackerResultsForGui  # type: ignore

# ---------------------------
# 1) Plug your config here
# ---------------------------
def build_eye_tracker():
    """
    Return a fully initialized EyeTracker with your config.
    You MUST edit this function to construct your EyeTrackerConfig
    (from brillouin_system.eye_tracker.eye_tracker_config import EyeTrackerConfig)
    and pass it to EyeTracker(...).

    Example sketch (adjust to your project):
        from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
        cfg = EyeTrackerConfig(
            binary_threshold_left=42,
            binary_threshold_right=42,
            frame_returned="original",          # or "binary" / "floodfilled"
            overlay_ellipse=True,
            do_ellipse_fitting=True,
            max_saving_freq_hz=30.0,
            save_images_path="session_data.h5",
        )
        return EyeTracker(cfg, use_dummy=False)

    If you want to test without cameras, you can pass use_dummy=True (if supported in your env),
    but you STILL need a valid EyeTrackerConfig object.
    """
    raise NotImplementedError("Implement build_eye_tracker() to return EyeTracker(config, ...).")


def np_rgb_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """
    Convert a HxWx3 uint8 RGB numpy array to QPixmap.
    Ensures a stable copy so the data doesn't get GC'ed.
    """
    if arr is None:
        return QPixmap()
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected RGB image with shape HxWx3.")

    h, w, _ = arr.shape
    # Ensure contiguous uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    # copy() to keep a deep copy managed by Qt
    return QPixmap.fromImage(qimg.copy())


class ImagePane(QLabel):
    """A QLabel that displays a numpy RGB image."""
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #111; color: #ddd; border: 1px solid #333;")
        self.setText(title)

    def show_np_image(self, img_rgb: np.ndarray | None, fallback_text: str = ""):
        if img_rgb is None:
            self.setText(fallback_text)
            return
        try:
            pix = np_rgb_to_qpixmap(img_rgb)
            if pix.isNull():
                self.setText(fallback_text)
            else:
                self.setPixmap(pix)
        except Exception as e:
            self.setText(f"{fallback_text}\n({e})")


class EyeTrackerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracker Viewer (Start/Stop)")
        self.tracker: EyeTracker | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 FPS (adjust as needed)
        self.timer.timeout.connect(self.update_frames)

        # 4 image panes
        self.left_cam_view = ImagePane("Left Camera")
        self.right_cam_view = ImagePane("Right Camera")
        self.rendered_view = ImagePane("Rendered Eye")
        self.xymap_view   = ImagePane("XY Map")

        # Buttons
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)

        # Layout: 2x2 grid for images
        grid = QGridLayout()
        grid.addWidget(self.left_cam_view, 0, 0)
        grid.addWidget(self.right_cam_view, 0, 1)
        grid.addWidget(self.rendered_view, 1, 0)
        grid.addWidget(self.xymap_view,   1, 1)

        # Controls
        controls = QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)

        root = QVBoxLayout()
        root.addLayout(grid)
        root.addLayout(controls)
        self.setLayout(root)
        self.resize(1100, 900)

    # ---- Start / Stop handlers ----
    def start_stream(self):
        if self.tracker is None:
            try:
                self.tracker = build_eye_tracker()
            except NotImplementedError:
                QMessageBox.critical(
                    self, "Config needed",
                    "Please edit build_eye_tracker() to create and return your EyeTracker instance."
                )
                return
            except Exception as e:
                QMessageBox.critical(self, "Failed to create EyeTracker", str(e))
                return

        self.timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_stream(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ---- Frame update ----
    def update_frames(self):
        if self.tracker is None:
            return
        try:
            res: EyeTrackerResultsForGui = self.tracker.get_display_frames()
            # Update 4 panes
            self.left_cam_view.show_np_image(res.cam_left_img, "Left Cam")
            self.right_cam_view.show_np_image(res.cam_right_img, "Right Cam")
            self.rendered_view.show_np_image(res.rendered_img, "Rendered Eye")
            self.xymap_view.show_np_image(res.xymap_img, "XY Map")
        except Exception as e:
            # Stop if something goes wrong, but keep UI alive
            self.stop_stream()
            QMessageBox.critical(self, "Stream error", f"{e}")

    # ---- Cleanup on close ----
    def closeEvent(self, event):
        try:
            self.stop_stream()
            # If your EyeTracker needs explicit shutdown, do it here, e.g.:
            # if self.tracker is not None:
            #     self.tracker.dual_cam_proxy.stop()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = EyeTrackerGUI()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
