# dual_camera_demo.py
import sys
import csv
import threading
import queue
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
    QFileDialog, QLineEdit, QShortcut
)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
import cv2
from PyQt5.QtWidgets import QCheckBox, QSpinBox, QLabel


# ⬇️ import your config dialog
from brillouin_system.devices.cameras.allied.allied_config.allied_config_dialog import AlliedConfigDialog
from brillouin_system.devices.cameras.allied.own_subprocess.dual_camera_proxy import DualCameraProxy
from brillouin_system.eye_tracker.stereo_calibration.detect_dot import detect_dot_with_blob
from brillouin_system.eye_tracker.stereo_calibration.init_stereo_cameras import stereo_cameras


def numpy_to_qpixmap_rgb(arr_rgb: np.ndarray) -> QPixmap:
    # arr_rgb: HxWx3, uint8, RGB
    h, w, _ = arr_rgb.shape
    if not arr_rgb.flags.c_contiguous:
        arr_rgb = np.ascontiguousarray(arr_rgb)
    bytes_per_line = int(arr_rgb.strides[0])
    qimg = QImage(arr_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def numpy_to_qpixmap_gray(arr: np.ndarray) -> QPixmap:
    # arr: HxW, expects 8-bit for display; shape already 2-D
    h, w = arr.shape
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    bytes_per_line = int(arr.strides[0])
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)


class DualCamImageCapture(QWidget):
    # Signal to update status text safely from non-GUI threads
    status_changed = pyqtSignal(str)

    def __init__(self, use_dummy=False):
        super().__init__()
        self.setWindowTitle("Dual Camera Shared-Memory Demo")

        # --- UI ---
        self.left_label = QLabel("Left")
        self.right_label = QLabel("Right")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.left_cfg_btn = QPushButton("Left Config")
        self.right_cfg_btn = QPushButton("Right Config")

        # Save pipeline UI
        self.save_pair_btn = QPushButton("Save Pair")
        self.choose_dir_btn = QPushButton("Choose Folder")
        self.save_status = QLabel("No folder selected")
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("filename prefix (optional)")
        self.prefix_input.setFixedWidth(180)

        # Images row
        top = QHBoxLayout()
        top.addWidget(self.left_label)
        top.addWidget(self.right_label)

        # Control buttons row
        btns = QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch()
        btns.addWidget(self.left_cfg_btn)
        btns.addWidget(self.right_cfg_btn)

        # Save controls row
        save_row = QHBoxLayout()
        save_row.addWidget(self.choose_dir_btn)
        save_row.addWidget(self.save_pair_btn)
        save_row.addWidget(self.prefix_input)
        save_row.addStretch()
        save_row.addWidget(self.save_status)

        # --- overlay controls ---
        overlay_row = QHBoxLayout()
        self.overlay_checkbox = QCheckBox("Overlay dot")
        self.overlay_checkbox.toggled.connect(self._clear_overlays)
        overlay_row.addWidget(self.overlay_checkbox)

        overlay_row.addWidget(QLabel("Cols:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 50)  # inner corners across
        self.cols_spin.setValue(10)  # common: 9x6
        overlay_row.addWidget(self.cols_spin)

        overlay_row.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 50)  # inner corners down
        self.rows_spin.setValue(8)
        overlay_row.addWidget(self.rows_spin)

        overlay_row.addStretch()


        # detection throttle
        self._detect_counter = 0
        self._last_overlay_left = None
        self._last_overlay_right = None

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(btns)
        layout.addLayout(save_row)
        layout.addLayout(overlay_row)
        self.setLayout(layout)

        # Proxy: start() attaches, then starts streaming
        self._setup_frame_views()
        self.proxy = DualCameraProxy(dtype="uint8", slots=8, use_dummy=use_dummy)
        self.proxy.start()

        # Frame handoff: reader thread -> main thread
        self._frame_q = queue.Queue(maxsize=2)
        self._reader_thread = None
        self._running = False

        # GUI timer to paint latest frame (non-blocking)
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.on_tick)

        # Wire buttons
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.left_cfg_btn.clicked.connect(self.open_left_config)
        self.right_cfg_btn.clicked.connect(self.open_right_config)
        self.choose_dir_btn.clicked.connect(self.on_choose_dir)
        self.save_pair_btn.clicked.connect(self.on_save_pair)

        # Thread-safe status updates
        self.status_changed.connect(self.save_status.setText)

        # Cache of last displayed frames
        self._last_left = None
        self._last_right = None
        self._last_ts = None

        # Save pipeline state
        self._save_dir: Path | None = None
        self._pair_idx = 0
        self._saver_q: queue.Queue = queue.Queue(maxsize=32)
        self._saver_running = True
        self._saver_thread = threading.Thread(target=self._saver_loop, daemon=True)
        self._saver_thread.start()

        # Keyboard shortcut: space bar to save
        QShortcut(QKeySequence("Space"), self, activated=self.on_save_pair)

        self._last_uv_left = None
        self._last_uv_right = None
        self.stereo = stereo_cameras


    def _set_pixmap_fit(self, label: QLabel, pm: QPixmap):
        # If the pixmap already matches the label geometry, avoid scaling work.
        if pm.width() == label.width() and pm.height() == label.height():
            label.setPixmap(pm)
            return
        # Only scale when dimensions differ, and prefer FastTransformation for speed.
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def _setup_frame_views(self):
        """
        Define fixed display areas for left/right, black background, and
        initialize with black placeholder pixmaps.
        """
        # Pick whatever slot size you want for your designed area
        slot_w, slot_h = 640, 480

        for lbl in (self.left_label, self.right_label):
            lbl.setFixedSize(slot_w, slot_h)  # fixed “slot” size
            lbl.setStyleSheet("background-color: black;")
            lbl.setAlignment(Qt.AlignCenter)

        # Set initial black placeholders (shown before Start)
        blank_left = QPixmap(self.left_label.size())
        blank_left.fill(Qt.black)
        self.left_label.setPixmap(blank_left)

        blank_right = QPixmap(self.right_label.size())
        blank_right.fill(Qt.black)
        self.right_label.setPixmap(blank_right)

    def _draw_dot_overlay(self, gray: np.ndarray) -> tuple[np.ndarray | None, tuple[float, float] | None]:
        """
        Detect a single dot in a grayscale frame and draw a small overlay for the UI.
        Returns (RGB_vis_or_None, (u, v)_or_None).
        """
        uv = detect_dot_with_blob(gray)
        if uv is None:
            return None, None

        u, v = uv
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # draw a small circle and crosshair
        cv2.circle(vis, (int(round(u)), int(round(v))), 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(round(u)), int(round(v))), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2, line_type=cv2.LINE_AA)
        # annotate with pixel coords
        cv2.putText(vis, f"({u:.1f}, {v:.1f})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return vis, (u, v)

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

        self._last_left = left
        self._last_right = right
        self._last_ts = ts

        if self.overlay_checkbox.isChecked():
            # refresh overlays occasionally to avoid UI stutter
            self._maybe_update_overlays(self._last_left, self._last_right)

            # left
            if self._last_overlay_left is not None:
                self._set_pixmap_fit(self.left_label, numpy_to_qpixmap_rgb(self._last_overlay_left))
            else:
                self._set_pixmap_fit(self.left_label, numpy_to_qpixmap_gray(self._last_left))

            # right
            if self._last_overlay_right is not None:
                self._set_pixmap_fit(self.right_label, numpy_to_qpixmap_rgb(self._last_overlay_right))
            else:
                self._set_pixmap_fit(self.right_label, numpy_to_qpixmap_gray(self._last_right))
        else:
            # plain grayscale
            self._set_pixmap_fit(self.left_label, numpy_to_qpixmap_gray(self._last_left))
            self._set_pixmap_fit(self.right_label, numpy_to_qpixmap_gray(self._last_right))

    # -------- Config dialogs --------
    def open_left_config(self):
        dlg = AlliedConfigDialog("left", self._apply_left_config, parent=self)
        dlg.exec_()

    def open_right_config(self):
        dlg = AlliedConfigDialog("right", self._apply_right_config, parent=self)
        dlg.exec_()

    def _apply_left_config(self, cfg_obj):
        was_running = self._running
        if was_running:
            self.on_stop()
        try:
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

    # -------- Save pipeline: UI handlers --------
    def on_choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not path:
            return
        self._init_save_dir(Path(path))

    def on_save_pair(self):
        if self._save_dir is None:
            self.status_changed.emit("Select a folder first")
            return
        if self._last_left is None or self._last_right is None:
            self.status_changed.emit("No frame yet")
            return

        prefix = self.prefix_input.text().strip()
        ts = self._last_ts
        idx = self._pair_idx  # snapshot index for this enqueue

        try:
            self._saver_q.put_nowait((
                idx,
                prefix,
                self._last_left,
                self._last_right,
                ts
            ))
            self.status_changed.emit(f"Queued pair {idx:04d}")
        except queue.Full:
            self.status_changed.emit("Save queue full—wait a moment")

    def _clear_overlays(self, *_):
        """Reset cached overlays when toggled or when we want a fresh compute."""
        self._last_overlay_left = None
        self._last_overlay_right = None
        self._detect_counter = 0

    def _maybe_update_overlays(self, left_gray: np.ndarray, right_gray: np.ndarray):
        """Throttled dot detection; refreshes overlay caches if found."""
        # run once every 3 frames; tweak to 2 or 5 depending on CPU
        self._detect_counter = (self._detect_counter + 1) % 3
        if self._detect_counter != 0:
            return

        try:
            visL, uvL = self._draw_dot_overlay(left_gray)
            visR, uvR = self._draw_dot_overlay(right_gray)

            self._last_overlay_left = visL
            self._last_overlay_right = visR

            # Store last detected pixel coords (for 3D if you want it)
            self._last_uv_left = uvL
            self._last_uv_right = uvR
        except Exception:
            self._last_overlay_left = None
            self._last_overlay_right = None
            self._last_uv_left = None
            self._last_uv_right = None
            visL, uvL = None, None
            visR, uvR = None, None

        # ------ OPTIONAL 3D: annotate world coordinates when both dots are present ------
        if self.stereo is not None and uvL is not None and uvR is not None and visL is not None and visR is not None:
            # Choose your preferred triangulation call; e.g., self.stereo.triangulate_best
            try:
                Xw = self.stereo.triangulate_linear(uvL, uvR)  # or triangulate_best(...) if you implemented it
                text = f"X=({Xw[0]:.2f}, {Xw[1]:.2f}, {Xw[2]:.2f})"
                for vis in (self._last_overlay_left, self._last_overlay_right):
                    cv2.putText(vis, text, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            except Exception:
                pass


    # -------- Save pipeline: setup --------
    def _init_save_dir(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        (root / "left").mkdir(exist_ok=True)
        (root / "right").mkdir(exist_ok=True)

        # Continue index if existing files are present
        existing = sorted((root / "left").glob("*.png"))
        if existing:
            def parse_idx(p: Path):
                # Expect something like 'pair_0012_left.png' or 'PREFIX_pair_0012_left.png'
                name = p.stem  # e.g., 'pair_0012_left'
                parts = name.split('_')
                # Find digits immediately before 'left'
                for i, token in enumerate(parts):
                    if token.isdigit() and i < len(parts) - 1 and parts[i + 1] == "left":
                        return int(token)
                # Fallback: highest numeric token in the name
                nums = [int(t) for t in parts if t.isdigit()]
                return max(nums) if nums else -1
            last = max((parse_idx(p) for p in existing), default=-1)
            self._pair_idx = last + 1
        else:
            self._pair_idx = 0

        self._save_dir = root
        self.status_changed.emit(f"Folder: {str(root)}  |  next #{self._pair_idx:04d}")

    # -------- Save pipeline: saver thread --------
    def _saver_loop(self):
        """
        Writes PNG pairs and appends a metadata CSV.
        Uses blocking get() so it sleeps when idle.
        """
        csv_file = None
        csv_writer = None
        csv_path = None

        while self._saver_running:
            try:
                idx, prefix, left, right, ts = self._saver_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if not self._saver_running:
                break

            if self._save_dir is None:
                # Nowhere to write; drop silently but keep running
                continue

            # Build filenames
            base = f"{prefix+'_' if prefix else ''}pair_{idx:04d}"
            left_path = self._save_dir / "left" / f"{base}_left.png"
            right_path = self._save_dir / "right" / f"{base}_right.png"

            # Write PNGs via Qt
            try:
                h, w = left.shape
                qimg_l = QImage(left.data, w, h, w, QImage.Format_Grayscale8)
                qimg_r = QImage(right.data, w, h, w, QImage.Format_Grayscale8)
                ok_l = qimg_l.save(str(left_path), "PNG")
                ok_r = qimg_r.save(str(right_path), "PNG")
                if not (ok_l and ok_r):
                    raise RuntimeError("QImage.save returned False")
            except Exception as e:
                print(f"[Saver] Failed to write images for pair {idx}: {e}")
                self.status_changed.emit(f"Save failed #{idx:04d}")
                continue

            # Prepare CSV writer (lazy open; add header if new)
            try:
                if csv_writer is None:
                    csv_path = self._save_dir / "metadata.csv"
                    file_exists = csv_path.exists()
                    csv_file = open(csv_path, "a", newline="")
                    csv_writer = csv.writer(csv_file)
                    if not file_exists:
                        csv_writer.writerow(["index", "timestamp", "left_path", "right_path"])
                        csv_file.flush()
                csv_writer.writerow([idx, ts, str(left_path), str(right_path)])
                csv_file.flush()
            except Exception as e:
                print(f"[Saver] Failed to write CSV for pair {idx}: {e}")
                # continue; images are already saved

            # Bump index for the next capture & update UI (via signal)
            self._pair_idx = idx + 1
            self.status_changed.emit(f"Saved #{idx:04d}  |  next #{self._pair_idx:04d}")

        # Cleanup
        try:
            if csv_file:
                csv_file.close()
        except Exception:
            pass

    def closeEvent(self, ev):
        try:
            self.on_stop()
        finally:
            try:
                # Stop saver thread
                self._saver_running = False
                # Unblock saver if waiting
                try:
                    self._saver_q.put_nowait((0, "", np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.uint8), 0))
                except queue.Full:
                    pass
                if self._saver_thread:
                    self._saver_thread.join(timeout=1.0)
            except Exception as e:
                print(f"[Demo] Saver shutdown error: {e}")
            self.proxy.shutdown()
        ev.accept()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # Windows

    app = QApplication(sys.argv)
    w = DualCamImageCapture(use_dummy=False)  # set False to use real hardware
    w.resize(1000, 560)
    w.show()
    sys.exit(app.exec_())
