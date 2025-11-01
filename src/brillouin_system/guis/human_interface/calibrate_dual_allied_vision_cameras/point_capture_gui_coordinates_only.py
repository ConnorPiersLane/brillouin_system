#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Camera GUI (coordinates-only variant)
- Removes frame/image saving pipeline completely.
- When "Save" is pressed:
    * Store (a) detected LEFT pixel coordinates (u,v) and
      (b) user-inserted coordinates (x,y,z) in memory and list widget.
    * If no detected LEFT coordinates are available, show a warning dialog.
- Show stored entries in a QListWidget.
- Provide "Remove Selected", "Remove All", and "Save All" (to JSON) buttons.
"""

import sys
import json
import queue
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QComboBox, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QShortcut
)
import cv2

# External project imports (unchanged from your environment)
from brillouin_system.devices.cameras.allied.allied_config.allied_config_dialog import AlliedConfigDialog
from brillouin_system.devices.cameras.allied.own_subprocess.dual_camera_proxy import DualCameraProxy
from brillouin_system.eye_tracker.stereo_imaging.se3 import SE3
from brillouin_system.eye_tracker.stereo_imaging.detect_dot import detect_dot_with_blob
from brillouin_system.eye_tracker.stereo_imaging.init_stereo_cameras import stereo_cameras


def numpy_to_qpixmap_rgb(arr_rgb: np.ndarray) -> QPixmap:
    if arr_rgb.dtype != np.uint8:
        arr_rgb = arr_rgb.astype(np.uint8, copy=False)
    if not arr_rgb.flags.c_contiguous:
        arr_rgb = np.ascontiguousarray(arr_rgb)
    h, w, c = arr_rgb.shape
    assert c == 3
    bytes_per_line = int(arr_rgb.strides[0])
    qimg = QImage(arr_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def numpy_to_qpixmap_gray(arr: np.ndarray) -> QPixmap:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    bytes_per_line = int(arr.strides[0])
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)


@dataclass
class CoordEntry:
    index: int
    timestamp: float
    left_uv: Tuple[float, float]            # (u, v) pixels in LEFT
    inserted_xyz: Tuple[float, float, float]  # (x, y, z) in current frame
    frame: str

    def to_display(self) -> str:
        u, v = self.left_uv
        x, y, z = self.inserted_xyz
        return f"#{self.index:04d} | ts={self.timestamp:.3f} | left=({u:.2f},{v:.2f}) | xyz=({x:.3f},{y:.3f},{z:.3f}) [{self.frame}]"


class DualCamCoordinateGUI(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self, use_dummy: bool = False):
        super().__init__()
        self.setWindowTitle("Dual Camera – Coordinates Only")

        # ==== Top image panes ====
        self.left_label = QLabel("Left")
        self.right_label = QLabel("Right")
        self._setup_frame_views()

        # ==== Primary controls ====
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.left_cfg_btn = QPushButton("Left Config")
        self.right_cfg_btn = QPushButton("Right Config")
        self.status_label = QLabel("")

        # Overlay + coordinate inputs
        self.overlay_checkbox = QPushButton("Overlay: OFF")
        self.overlay_checkbox.setCheckable(True)
        self.overlay_checkbox.toggled.connect(self._clear_overlays)

        self.x_input = QLineEdit(); self.x_input.setFixedWidth(100); self.x_input.setPlaceholderText("x")
        self.y_input = QLineEdit(); self.y_input.setFixedWidth(100); self.y_input.setPlaceholderText("y")
        self.z_input = QLineEdit(); self.z_input.setFixedWidth(100); self.z_input.setPlaceholderText("z")

        self.load_tf_btn = QPushButton("Load Ref (JSON)")
        self.load_tf_btn.clicked.connect(self.on_load_transform)
        self.frame_combo = QComboBox(); self.frame_combo.addItem("left")

        # Save single entry (coordinates) controls
        self.save_btn = QPushButton("Save Entry")

        # List + batch controls
        self.list_widget = QListWidget()
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_all_btn = QPushButton("Remove All")
        self.save_all_btn = QPushButton("Save All (JSON)")

        # ==== Layout ====
        top = QHBoxLayout(); top.addWidget(self.left_label); top.addWidget(self.right_label)

        ctl = QHBoxLayout()
        ctl.addWidget(self.start_btn); ctl.addWidget(self.stop_btn); ctl.addStretch()
        ctl.addWidget(self.left_cfg_btn); ctl.addWidget(self.right_cfg_btn)

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(self.overlay_checkbox)
        overlay_row.addSpacing(10)
        overlay_row.addWidget(QLabel("X:")); overlay_row.addWidget(self.x_input)
        overlay_row.addWidget(QLabel("Y:")); overlay_row.addWidget(self.y_input)
        overlay_row.addWidget(QLabel("Z:")); overlay_row.addWidget(self.z_input)
        overlay_row.addStretch()
        overlay_row.addWidget(self.load_tf_btn)
        overlay_row.addWidget(QLabel("Frame:"))
        overlay_row.addWidget(self.frame_combo)
        overlay_row.addSpacing(12)
        overlay_row.addWidget(self.save_btn)
        overlay_row.addStretch()
        overlay_row.addWidget(self.status_label)

        list_row = QHBoxLayout()
        list_row.addWidget(self.list_widget)

        btn_col = QVBoxLayout()
        btn_col.addWidget(self.remove_selected_btn)
        btn_col.addWidget(self.remove_all_btn)
        btn_col.addStretch()
        btn_col.addWidget(self.save_all_btn)
        list_row.addLayout(btn_col)

        lay = QVBoxLayout()
        lay.addLayout(top)
        lay.addLayout(ctl)
        lay.addLayout(overlay_row)
        lay.addLayout(list_row)
        self.setLayout(lay)

        # ==== Camera / threads ====
        self.proxy = DualCameraProxy(dtype="uint8", slots=8, use_dummy=use_dummy)
        self.proxy.start()
        self._frame_q: "queue.Queue[tuple[np.ndarray, np.ndarray, float]]" = queue.Queue(maxsize=2)
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

        self.timer = QTimer(self); self.timer.setInterval(40); self.timer.timeout.connect(self.on_tick)

        # ==== Internal state ====
        self._last_left: Optional[np.ndarray] = None
        self._last_right: Optional[np.ndarray] = None
        self._last_ts: Optional[float] = None

        self._last_overlay_left = None
        self._last_overlay_right = None
        self._last_uv_left: Optional[Tuple[float,float]] = None
        self._last_uv_right: Optional[Tuple[float,float]] = None

        self._detect_counter = 0
        self.entries: List[CoordEntry] = []
        self._next_index = 0

        # Transforms
        self.transforms: dict[str, SE3] = {"left": SE3(np.eye(3), np.zeros(3))}
        self.stereo = stereo_cameras

        # ==== Wiring ====
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.left_cfg_btn.clicked.connect(self.open_left_config)
        self.right_cfg_btn.clicked.connect(self.open_right_config)

        self.save_btn.clicked.connect(self.on_save_single)
        self.remove_selected_btn.clicked.connect(self.on_remove_selected)
        self.remove_all_btn.clicked.connect(self.on_remove_all)
        self.save_all_btn.clicked.connect(self.on_save_all_json)

        self.overlay_checkbox.toggled.connect(self._update_overlay_button_text)

        # Spacebar to save single entry
        QShortcut(QKeySequence("Space"), self, activated=self.on_save_single)

        # Status updates
        self.status_changed.connect(self.status_label.setText)

    # ====== UI helpers ======
    def _setup_frame_views(self):
        slot_w, slot_h = 640, 480
        for lbl in (self.left_label, self.right_label):
            lbl.setFixedSize(slot_w, slot_h)
            lbl.setStyleSheet("background-color: black;")
            lbl.setAlignment(Qt.AlignCenter)
            pm = QPixmap(lbl.size()); pm.fill(Qt.black); lbl.setPixmap(pm)

    def _set_pixmap_fit(self, label: QLabel, pm: QPixmap):
        if pm.width() == label.width() and pm.height() == label.height():
            label.setPixmap(pm); return
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def _update_overlay_button_text(self, checked: bool):
        self.overlay_checkbox.setText("Overlay: ON" if checked else "Overlay: OFF")

    def _clear_overlays(self, *_):
        self._last_overlay_left = None
        self._last_overlay_right = None
        self._detect_counter = 0

    # ====== Threads / painting ======
    def _reader_loop(self):
        while self._running:
            try:
                left, right, ts = self.proxy.get_frames()
                if not self._frame_q.empty():
                    try: self._frame_q.get_nowait()
                    except queue.Empty: pass
                self._frame_q.put((left, right, ts))
            except Exception as e:
                print(f"[Reader] {e}")
                break

    def on_start(self):
        if self._running: return
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
        self._last_left, self._last_right, self._last_ts = left, right, ts

        if self.overlay_checkbox.isChecked():
            self._maybe_update_overlays(left, right)
            if self._last_overlay_left is not None:
                self._set_pixmap_fit(self.left_label, numpy_to_qpixmap_rgb(self._last_overlay_left))
            else:
                self._set_pixmap_fit(self.left_label, numpy_to_qpixmap_gray(left))
            if self._last_overlay_right is not None:
                self._set_pixmap_fit(self.right_label, numpy_to_qpixmap_rgb(self._last_overlay_right))
            else:
                self._set_pixmap_fit(self.right_label, numpy_to_qpixmap_gray(right))
        else:
            self._set_pixmap_fit(self.left_label, numpy_to_qpixmap_gray(left))
            self._set_pixmap_fit(self.right_label, numpy_to_qpixmap_gray(right))

    def _draw_dot_overlay(self, gray: np.ndarray):
        uv = detect_dot_with_blob(gray)
        if uv is None:
            return None, None
        u, v = uv
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.circle(vis, (int(round(u)), int(round(v))), 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(round(u)), int(round(v))), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2, line_type=cv2.LINE_AA)
        cv2.putText(vis, f"({u:.1f}, {v:.1f})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return vis, (u, v)

    def _maybe_update_overlays(self, left_gray: np.ndarray, right_gray: np.ndarray):
        self._detect_counter = (self._detect_counter + 1) % 3
        if self._detect_counter != 0: return
        try:
            visL, uvL = self._draw_dot_overlay(left_gray)
            visR, uvR = self._draw_dot_overlay(right_gray)
            self._last_overlay_left = visL
            self._last_overlay_right = visR
            self._last_uv_left = uvL
            self._last_uv_right = uvR
        except Exception:
            self._last_overlay_left = None
            self._last_overlay_right = None
            self._last_uv_left = None
            self._last_uv_right = None
            visL = visR = None; uvL = uvR = None

        if self.stereo is not None and uvL is not None and uvR is not None and visL is not None and visR is not None:
            try:
                X_left = self.stereo.triangulate_best(uvL, uvR)
                X_out = self._to_current_frame(X_left)
                text = f"X=({X_out[0]:.2f}, {X_out[1]:.2f}, {X_out[2]:.2f}) [{self.frame_combo.currentText()}]"
                for vis in (self._last_overlay_left, self._last_overlay_right):
                    cv2.putText(vis, text, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            except Exception:
                pass

    def _to_current_frame(self, X_left: np.ndarray) -> np.ndarray:
        frame = self.frame_combo.currentText()
        T = self.transforms.get(frame)
        if T is None: return X_left
        return T.apply_points(X_left)

    def on_load_transform(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Reference Transform (JSON)", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            R = np.asarray(d["R"], float).reshape(3, 3)
            t = np.asarray(d["t"], float).reshape(3)
            T_left_to_ref = SE3(R, t)
        except Exception as e:
            QMessageBox.warning(self, "Load Transform", f"Failed to load transform: {e}")
            return
        name = Path(path).stem
        base = name; k = 1
        while name in self.transforms:
            name = f"{base}_{k}"; k += 1
        self.transforms[name] = T_left_to_ref
        self.frame_combo.addItem(name)
        self.status_changed.emit(f"Loaded reference '{name}'")

    # ====== Coordinate capture & list management ======
    def _read_xyz_inputs(self) -> Optional[Tuple[float,float,float]]:
        def to_float(s):
            s = (s or "").strip()
            try: return float(s)
            except ValueError: return None
        x = to_float(self.x_input.text())
        y = to_float(self.y_input.text())
        z = to_float(self.z_input.text())
        if x is None or y is None or z is None:
            return None
        return (x, y, z)

    def on_save_single(self):
        # Require detected LEFT (u,v)
        if self._last_uv_left is None or self._last_ts is None:
            QMessageBox.warning(self, "No point detected", "No left coordinates detected. Make sure the dot is visible and overlay is ON.")
            return
        ins = self._read_xyz_inputs()
        if ins is None:
            QMessageBox.warning(self, "Missing XYZ", "Please enter valid numeric X, Y, Z values before saving.")
            return
        entry = CoordEntry(
            index=self._next_index,
            timestamp=float(self._last_ts),
            left_uv=(float(self._last_uv_left[0]), float(self._last_uv_left[1])),
            inserted_xyz=(float(ins[0]), float(ins[1]), float(ins[2])),
            frame=self.frame_combo.currentText(),
        )
        self.entries.append(entry)
        self._next_index += 1

        item = QListWidgetItem(entry.to_display())
        item.setData(Qt.UserRole, entry)  # stash the full entry
        self.list_widget.addItem(item)
        self.status_changed.emit(f"Saved entry #{entry.index:04d}")

    def on_remove_selected(self):
        rows = sorted([i.row() for i in self.list_widget.selectedIndexes()], reverse=True)
        for row in rows:
            self.list_widget.takeItem(row)
            if 0 <= row < len(self.entries):
                self.entries.pop(row)
        # Reindex for display consistency
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            e: CoordEntry = item.data(Qt.UserRole)
            e.index = i
            item.setText(e.to_display())
        self._next_index = self.list_widget.count()

    def on_remove_all(self):
        self.list_widget.clear()
        self.entries.clear()
        self._next_index = 0
        self.status_changed.emit("Cleared all entries")

    def on_save_all_json(self):
        if not self.entries:
            QMessageBox.information(self, "Nothing to save", "There are no entries to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Coordinates JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            payload = {
                "count": len(self.entries),
                "entries": [asdict(e) for e in self.entries],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save JSON:\n{e}")
            return
        QMessageBox.information(self, "Saved", f"Saved {len(self.entries)} entries to:\n{path}")

    # ====== Config dialogs ======
    def open_left_config(self):
        dlg = AlliedConfigDialog("left", self._apply_left_config, parent=self)
        dlg.exec_()

    def open_right_config(self):
        dlg = AlliedConfigDialog("right", self._apply_right_config, parent=self)
        dlg.exec_()

    def _apply_left_config(self, cfg_obj):
        was_running = self._running
        if was_running: self.on_stop()
        try:
            self.proxy.set_configs(cfg_left=cfg_obj, cfg_right=None)
            print("[GUI] Left config applied.")
        finally:
            if was_running: self.on_start()

    def _apply_right_config(self, cfg_obj):
        was_running = self._running
        if was_running: self.on_stop()
        try:
            self.proxy.set_configs(cfg_left=None, cfg_right=cfg_obj)
            print("[GUI] Right config applied.")
        finally:
            if was_running: self.on_start()

    # ====== Shutdown ======
    def closeEvent(self, ev):
        try:
            self.on_stop()
            self.proxy.shutdown()
        except Exception:
            pass
        ev.accept()


def main():
    import multiprocessing as mp
    mp.freeze_support()
    app = QApplication(sys.argv)
    w = DualCamCoordinateGUI(use_dummy=True)  # set False to use real hardware
    w.resize(1000, 600)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
