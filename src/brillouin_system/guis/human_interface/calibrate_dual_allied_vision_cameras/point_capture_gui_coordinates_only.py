
import sys
import csv
import threading
import queue
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
    QFileDialog, QLineEdit, QShortcut, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
import cv2
from PyQt5.QtWidgets import QCheckBox, QSpinBox, QLabel


# ⬇️ import your config dialog
from brillouin_system.devices.cameras.allied.allied_config.allied_config_dialog import AlliedConfigDialog
from brillouin_system.devices.cameras.allied.own_subprocess.dual_camera_proxy import DualCameraProxy
from brillouin_system.eye_tracker.stereo_imaging.fit_coordinate_system import fit_coordinate_system
from brillouin_system.eye_tracker.stereo_imaging.se3 import SE3
from brillouin_system.eye_tracker.stereo_imaging.detect_dot import detect_dot_with_blob, detect_dot_with_blob_dummy
from brillouin_system.eye_tracker.stereo_imaging.init_stereo_cameras import stereo_cameras


def numpy_to_qpixmap_rgb(arr_rgb: np.ndarray) -> QPixmap:
    # arr_rgb: HxWx3, dtype=uint8, RGB
    if arr_rgb.dtype != np.uint8:
        arr_rgb = arr_rgb.astype(np.uint8, copy=False)
    if not arr_rgb.flags.c_contiguous:
        arr_rgb = np.ascontiguousarray(arr_rgb)

    h, w, c = arr_rgb.shape
    assert c == 3
    bytes_per_line = int(arr_rgb.strides[0])  # ✅ NOT 3*w, use stride
    qimg = QImage(arr_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def numpy_to_qpixmap_gray(arr: np.ndarray) -> QPixmap:
    # arr: HxW, dtype=uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)

    h, w = arr.shape
    bytes_per_line = int(arr.strides[0])  # ✅ true row stride
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

        # ⬇️ NEW: a list widget to show saved pairs
        from PyQt5.QtWidgets import QListWidget
        self.pair_list = QListWidget()

        # Save (now just "capture coordinates") UI
        self.save_pair_btn = QPushButton("Save")
        self.save_status = QLabel("Ready")
        self.prefix_input = QLineEdit()  # kept if you still want a label/prefix column
        self.prefix_input.setPlaceholderText("optional label")
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

        # Save controls row (no folder now)
        save_row = QHBoxLayout()
        save_row.addWidget(self.save_pair_btn)
        save_row.addWidget(self.prefix_input)
        save_row.addStretch()
        save_row.addWidget(self.save_status)

        # --- overlay controls ---
        overlay_row = QHBoxLayout()
        self.overlay_checkbox = QCheckBox("Overlay dot")
        self.overlay_checkbox.toggled.connect(self._clear_overlays)
        overlay_row.addWidget(self.overlay_checkbox)

        overlay_row.addWidget(QLabel("X:"))
        self.x_input = QLineEdit();
        self.x_input.setFixedWidth(100);
        self.x_input.setPlaceholderText("x")
        overlay_row.addWidget(self.x_input)

        overlay_row.addWidget(QLabel("Y:"))
        self.y_input = QLineEdit();
        self.y_input.setFixedWidth(100);
        self.y_input.setPlaceholderText("y")
        overlay_row.addWidget(self.y_input)

        overlay_row.addWidget(QLabel("Z:"))
        self.z_input = QLineEdit();
        self.z_input.setFixedWidth(100);
        self.z_input.setPlaceholderText("z")
        overlay_row.addWidget(self.z_input)

        overlay_row.addStretch()

        # --- reference frames (output coordinate system) ---
        self.load_tf_btn = QPushButton("Load Ref (JSON)")
        self.load_tf_btn.clicked.connect(self.on_load_transform)
        overlay_row.addWidget(self.load_tf_btn)

        overlay_row.addWidget(QLabel("Output frame:"))
        self.frame_combo = QComboBox()
        self.frame_combo.addItem("left")  # default: stereo world = left camera frame
        overlay_row.addWidget(self.frame_combo)

        # Coordinate-system registry: maps frame name -> SE3 (left->frame)
        self.transforms: dict[str, SE3] = {
            "left": SE3(np.eye(3), np.zeros(3)),  # identity
        }

        # detection throttle
        self._detect_counter = 0
        self._last_overlay_left = None
        self._last_overlay_right = None

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(btns)
        layout.addLayout(save_row)
        layout.addLayout(overlay_row)
        # ⬇️ NEW: add the list widget to the main layout
        layout.addWidget(self.pair_list)

        # ⬇️ NEW: buttons below the list widget
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_all_btn = QPushButton("Remove All")
        self.calc_transform_btn = QPushButton("Calculate Transform")  # ⬅️ NEW

        remove_row = QHBoxLayout()
        remove_row.addWidget(self.remove_selected_btn)
        remove_row.addWidget(self.remove_all_btn)
        remove_row.addStretch()
        remove_row.addWidget(self.calc_transform_btn)  # ⬅️ NEW (placed at right)
        layout.addLayout(remove_row)

        # connect signals
        self.remove_selected_btn.clicked.connect(self.on_remove_selected)
        self.remove_all_btn.clicked.connect(self.on_remove_all)
        self.calc_transform_btn.clicked.connect(self.on_calculate_transform)  # ⬅️ NEW

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
        self.save_pair_btn.clicked.connect(self.on_save_pair)

        # Thread-safe status updates
        self.status_changed.connect(self.save_status.setText)

        # Cache of last displayed frames
        self._last_left = None
        self._last_right = None
        self._last_ts = None

        # Keyboard shortcut: space bar to save
        QShortcut(QKeySequence("Space"), self, activated=self.on_save_pair)

        # Last detected UV and 3D points
        self._last_uv_left = None
        self._last_uv_right = None
        self._last_X_left = None  # ⬅️ NEW: store last triangulated 3D in LEFT
        self._last_X_out = None  # (optional) current output frame version

        self.stereo = stereo_cameras

        # ⬇️ NEW: in-memory “datasets”
        self.stereo_points_left = []  # list of [x, y, z] from triangulation (LEFT frame)
        self.typed_points = []  # list of [x, y, z] from inputs

    def on_calculate_transform(self):
        """
        Fit a transform from LEFT -> (current output frame) using the in-memory pairs:
          self.stereo_points_left  (Nx3, LEFT)
          self.typed_points        (Nx3, target)
        Prompts to save JSON if the fit succeeds.
        """
        import numpy as np
        import sys, os, traceback
        from PyQt5.QtWidgets import QMessageBox

        # --- Gather valid pairs
        A, B = [], []
        for left_xyz, typed_xyz in zip(self.stereo_points_left, self.typed_points):
            if (left_xyz is None) or (typed_xyz is None):
                continue
            if (len(left_xyz) != 3) or (len(typed_xyz) != 3):
                continue
            if any(v is None for v in left_xyz) or any(v is None for v in typed_xyz):
                continue
            try:
                A.append([float(left_xyz[0]), float(left_xyz[1]), float(left_xyz[2])])
                B.append([float(typed_xyz[0]), float(typed_xyz[1]), float(typed_xyz[2])])
            except Exception:
                continue

        if len(A) < 3:
            msg = "Need at least 3 valid LEFT↔INPUT pairs (both sides filled) to compute transform."
            self.status_changed.emit(msg)
            QMessageBox.information(self, "Not enough pairs", msg)
            return

        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        # --- Import fitter (be liberal about sys.path so it works from anywhere)
        try:
            # Add script directory to sys.path if needed
            here = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
            if here and here not in sys.path:
                sys.path.insert(0, here)
            # Add cwd as well (common dev case)
            cwd = os.getcwd()
            if cwd and cwd not in sys.path:
                sys.path.insert(0, cwd)


        except Exception as e:
            tb = traceback.format_exc()
            self.status_changed.emit(f"Cannot import fit_coordinate_system: {e}")
            print("[Transform] Import error:\n", tb)
            QMessageBox.critical(self, "Import error", f"Cannot import fit_coordinate_system:\n{e}")
            return

        # --- Run fit (no scaling by default)
        with_scale = False
        trim_fraction = 0.0
        trim_repeats = 1

        try:
            result, info = fit_coordinate_system(
                points_left=A,
                points_zaber=B,
                with_scale=with_scale,
                trim_fraction=trim_fraction,
                trim_repeats=trim_repeats,
            )
        except Exception as e:
            tb = traceback.format_exc()
            self.status_changed.emit(f"Fit failed: {e}")
            print("[Transform] Fit failed:\n", tb)
            QMessageBox.critical(self, "Fit failed", f"{e}")
            return

        # --- Normalize result to (R, t)
        R, t = None, None
        try:
            # Case 1: result is an SE3-like object with .R and .t
            if hasattr(result, "R") and hasattr(result, "t"):
                R = np.asarray(result.R, float).reshape(3, 3)
                t = np.asarray(result.t, float).reshape(3)
            else:
                # Case 2: result is (R, t)
                R, t = result
                R = np.asarray(R, float).reshape(3, 3)
                t = np.asarray(t, float).reshape(3)
        except Exception as e:
            tb = traceback.format_exc()
            self.status_changed.emit(f"Unexpected fit result shape: {e}")
            print("[Transform] Unexpected result:\n", tb)
            QMessageBox.critical(self, "Unexpected result", f"Unexpected fit result:\n{e}")
            return

        rms = float(info.get("rms", float("nan"))) if isinstance(info, dict) else float("nan")
        scale_val = float(info.get("scale", 1.0)) if isinstance(info, dict) else 1.0
        print(f"[Transform] Fitted. N={len(A)}, RMS={rms:.6f}, scale={scale_val:.6f}")

        # --- Build JSON
        target_name = self.frame_combo.currentText() or "target"
        json_obj = self._build_transform_json(
            name=f"left_to_{target_name}",
            description=f"Transformation from left camera frame to {target_name} coordinate system",
            R=R,
            t=t,
            source_info={
                "method": "umeyama_fit",
                "num_points": int(len(A)),
                "rms_error": rms,
                "with_scale": bool(abs(scale_val - 1.0) > 1e-12),
                "scale": scale_val,
            }
        )

        # --- Ask where to save
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transform JSON",
            f"left_to_{target_name}.json",
            "JSON (*.json)"
        )
        if not path:
            self.status_changed.emit("Save canceled")
            return

        # --- Write JSON & optionally register in UI
        try:
            import json, datetime
            json_obj["created_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2)

            self.status_changed.emit(f"Saved transform to: {path}")
            QMessageBox.information(self, "Transform saved", f"Saved JSON:\n{path}")

            # Also add it to the frames dropdown for immediate use
            try:
                T_loaded = SE3(R, t)
                name = Path(path).stem
                base = name
                k = 1
                while name in self.transforms:
                    name = f"{base}_{k}"
                    k += 1
                self.transforms[name] = T_loaded
                self.frame_combo.addItem(name)
            except Exception:
                pass

        except Exception as e:
            tb = traceback.format_exc()
            self.status_changed.emit(f"Failed to write JSON: {e}")
            print("[Transform] Write error:\n", tb)
            QMessageBox.critical(self, "Write error", f"{e}")

    def _build_transform_json(self, *, name: str, description: str, R, t, source_info: dict) -> dict:
        """
        Build a JSON payload similar to the provided example file:
        {
          "name": "...",
          "description": "...",
          "R": [[...],[...],[...]],
          "t": [x,y,z],
          "created_at": "...",
          "source": { ... }
        }
        """
        import numpy as np
        R = np.asarray(R, float).reshape(3, 3)
        t = np.asarray(t, float).reshape(3)

        obj = {
            "name": name,
            "description": description,
            "R": R.tolist(),
            "t": t.tolist(),
            # created_at added at save-time in on_calculate_transform()
            "source": source_info or {},
        }
        return obj

    def _read_xyz_inputs(self):
        def to_float(s):
            s = (s or "").strip()
            try:
                return float(s)
            except ValueError:
                return None

        x = to_float(self.x_input.text())
        y = to_float(self.y_input.text())
        z = to_float(self.z_input.text())
        # Return None if any is missing/invalid; we’ll still save images.
        return (x, y, z) if (x is not None and y is not None and z is not None) else None

    def _set_pixmap_fit(self, label: QLabel, pm: QPixmap):
        # If the pixmap already matches the label geometry, avoid scaling work.
        if pm.width() == label.width() and pm.height() == label.height():
            label.setPixmap(pm)
            return
        # Only scale when dimensions differ, and prefer FastTransformation for speed.
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def on_remove_selected(self):
        """Remove the currently selected row(s) from both the list widget and internal arrays."""
        selected = self.pair_list.selectedIndexes()
        if not selected:
            self.status_changed.emit("No item selected")
            return

        # Collect indices in reverse order to remove safely
        rows = sorted((s.row() for s in selected), reverse=True)

        for row in rows:
            try:
                del self.stereo_points_left[row]
                del self.typed_points[row]
                self.pair_list.takeItem(row)
            except IndexError:
                continue

        self.status_changed.emit(f"Removed {len(rows)} selected item(s)")
        self._refresh_list_indices()

    def on_remove_all(self):
        """Clear all stored data and reset the list."""
        count = len(self.stereo_points_left)
        self.stereo_points_left.clear()
        self.typed_points.clear()
        self.pair_list.clear()
        self.status_changed.emit(f"Cleared all {count} item(s)")

    def _refresh_list_indices(self):
        """Re-number list widget items after deletions."""
        for idx in range(self.pair_list.count()):
            item = self.pair_list.item(idx)
            text = item.text()
            # Replace the leading "[0000]" index with the new one
            new_text = f"[{idx:04d}]" + text[text.find("]") + 1:]
            item.setText(new_text)

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


    def on_save_pair(self):
        """
        Capture the current stereo 3D point (LEFT frame) and the typed XYZ,
        store them into two parallel lists-of-lists, and render a row in the QListWidget.
        """
        # Read the latest triangulated point from the overlay pipeline
        X_left = self._last_X_left  # may be None if no good detection yet
        typed = self._read_xyz_inputs()  # tuple or None

        # We allow saving even if one side is missing; store None as blank for visibility.
        left_xyz = [None, None, None]
        if X_left is not None and len(X_left) == 3:
            left_xyz = [float(X_left[0]), float(X_left[1]), float(X_left[2])]

        typed_xyz = [None, None, None]
        if typed is not None:
            typed_xyz = [float(typed[0]), float(typed[1]), float(typed[2])]

        # Append to in-memory datasets
        self.stereo_points_left.append(left_xyz)
        self.typed_points.append(typed_xyz)

        # Optional label the user typed (prefix_input reused as a free-form note)
        label = self.prefix_input.text().strip()

        # Render a compact row in the list widget
        self._append_pair_row(label, left_xyz, typed_xyz)

        # Status
        idx = len(self.stereo_points_left) - 1
        self.status_changed.emit(f"Stored pair #{idx:04d}")

    def _append_pair_row(self, label: str, left_xyz: list, typed_xyz: list):
        """
        Adds a human-readable line into self.pair_list, showing:
          [idx] <label> | LEFT: (x,y,z) | INPUT: (x,y,z)
        """
        from PyQt5.QtWidgets import QListWidgetItem
        idx = len(self.stereo_points_left) - 1

        def fmt(v):
            return "None" if v is None else f"{v:.3f}"

        left_txt = f"({fmt(left_xyz[0])}, {fmt(left_xyz[1])}, {fmt(left_xyz[2])})"
        typed_txt = f"({fmt(typed_xyz[0])}, {fmt(typed_xyz[1])}, {fmt(typed_xyz[2])})"
        text = f"[{idx:04d}] {label or ''} | LEFT: {left_txt} | INPUT: {typed_txt}"
        self.pair_list.addItem(QListWidgetItem(text))

    def _clear_overlays(self, *_):
        """Reset cached overlays when toggled or when we want a fresh compute."""
        self._last_overlay_left = None
        self._last_overlay_right = None
        self._detect_counter = 0

    def _maybe_update_overlays(self, left_gray: np.ndarray, right_gray: np.ndarray):
        self._detect_counter = (self._detect_counter + 1) % 3
        if self._detect_counter != 0:
            return

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
            visL, uvL = None, None
            visR, uvR = None, None

        # ⬇️ Triangulate and cache the last 3D point in LEFT (and current frame)
        self._last_X_left = None
        self._last_X_out = None
        if self.stereo is not None and uvL is not None and uvR is not None and visL is not None and visR is not None:
            try:
                X_left = self.stereo.triangulate_linear(uvL, uvR)  # 3D in LEFT
                self._last_X_left = X_left.astype(float).reshape(3)
                X_out = self._to_current_frame(self._last_X_left)  # map to selected frame
                self._last_X_out = X_out.astype(float).reshape(3)
                text = f"X=({X_out[0]:.2f}, {X_out[1]:.2f}, {X_out[2]:.2f}) [{self.frame_combo.currentText()}]"
                for vis in (self._last_overlay_left, self._last_overlay_right):
                    cv2.putText(vis, text, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            except Exception:
                # if triangulation fails, keep overlays but no 3D reading
                self._last_X_left = None
                self._last_X_out = None
                pass

    def _to_current_frame(self, X_left: np.ndarray) -> np.ndarray:
        """Map 3D point from LEFT frame to the currently selected output frame."""
        frame = self.frame_combo.currentText()
        T = self.transforms.get(frame)
        if T is None:
            # Fallback to left if something odd happens
            return X_left
        return T.apply_points(X_left)

    def on_load_transform(self):
        # Expect a JSON with {"R": [[...],[...],[...]], "t": [x,y,z]}
        path, _ = QFileDialog.getOpenFileName(self, "Load Reference Transform (JSON)", "", "JSON (*.json)")
        if not path:
            return
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            R = np.asarray(d["R"], float).reshape(3, 3)
            t = np.asarray(d["t"], float).reshape(3)
            T_left_to_ref = SE3(R, t)
        except Exception as e:
            self.status_changed.emit(f"Failed to load transform: {e}")
            return

        # Use file stem as name; avoid collisions
        name = Path(path).stem
        base = name
        k = 1
        while name in self.transforms:
            name = f"{base}_{k}"
            k += 1

        self.transforms[name] = T_left_to_ref
        self.frame_combo.addItem(name)
        self.status_changed.emit(f"Loaded reference '{name}'")


    def closeEvent(self, ev):
        try:
            self.on_stop()
        finally:
            try:
                # Stop saver thread
                self._saver_running = False
                # Unblock saver if waiting
                try:
                    self._saver_q.put_nowait(
                        (0, "", np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.uint8), 0, None, "left"))
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
