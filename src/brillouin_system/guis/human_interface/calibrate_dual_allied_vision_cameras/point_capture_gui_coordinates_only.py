"""
Point-capture GUI for the LEFT-camera -> Zaber coordinate transform.

Workflow:
  1. Mount the calibration dot on the Zaber stage, start the cameras.
  2. Move the stage to a pose; the GUI detects the dot in both cameras,
     triangulates it (LEFT frame, mm) and shows the reprojection RMS live.
  3. Press Save (Space): the capture averages the last ~2 s of detections,
     and the Zaber position is read automatically (if connected) or taken
     from the X/Y/Z fields.
  4. Repeat over the working volume. The transform is re-fitted after every
     capture; per-point residuals, RMS, a units sanity check and geometry
     hints update live.
  5. "Install as active" writes stereo_configs/left_to_zaber.json directly
     (with a timestamped backup of the previous file).

Every capture is auto-saved to a session file so a crash loses nothing.
"""

import sys
import json
import time
import datetime
import shutil
import threading
import queue
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
    QFileDialog, QLineEdit, QShortcut, QCheckBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QGroupBox, QMessageBox, QHeaderView,
)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QColor
from PyQt5.QtCore import QTimer, pyqtSignal, Qt

from brillouin_system.devices.cameras.allied.allied_config.allied_config_dialog import AlliedConfigDialog
from brillouin_system.devices.cameras.allied.own_subprocess.dual_camera_proxy import DualCameraProxy
from brillouin_system.eye_tracker.stereo_imaging.fit_coordinate_system import fit_coordinate_system
from brillouin_system.eye_tracker.stereo_imaging.se3 import SE3
from brillouin_system.eye_tracker.stereo_imaging.detect_dot import detect_dot, make_blob_detector
from brillouin_system.eye_tracker.stereo_imaging.init_stereo_cameras import stereo_cameras

import brillouin_system.eye_tracker.stereo_imaging as _stereo_imaging_pkg

STEREO_CONFIGS_DIR = Path(_stereo_imaging_pkg.__file__).resolve().parent / "stereo_configs"
ACTIVE_TRANSFORM_PATH = STEREO_CONFIGS_DIR / "left_to_zaber.json"
SESSION_PATH = Path(__file__).resolve().parent / "point_capture_session.json"

TRI_RMS_WARN_PX = 2.0        # triangulation reprojection RMS above this = suspicious detection
SCALE_WARN = 0.02            # |scale-1| above this suggests a units mismatch
CAPTURE_WINDOW_S = 2.0       # average detections over this window on capture
CAPTURE_MIN_SAMPLES = 5


# ---------------- pure helpers (testable without Qt) ----------------
def fit_pairs(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Rigid Umeyama fit LEFT->ZABER plus diagnostics.
    Returns dict with T (SE3), rms, residuals (N,), scale (from a separate
    similarity fit — should be ~1.0 if LEFT and ZABER use the same units).
    """
    T, info = fit_coordinate_system(points_left=A, points_zaber=B, with_scale=False)
    pred = T.apply_points(A)
    residuals = np.linalg.norm(pred - B, axis=1)
    try:
        _, info_s = fit_coordinate_system(points_left=A, points_zaber=B, with_scale=True)
        scale = float(info_s["scale"])
    except Exception:
        scale = float("nan")
    return {"T": T, "rms": float(info["rms"]), "residuals": residuals, "scale": scale}


def motion_check(prev_left, prev_zaber, left, zaber, R: np.ndarray | None = None,
                 mag_tol: float = 0.2, ang_tol_deg: float = 15.0,
                 min_step: float = 0.05) -> tuple[str, bool]:
    """
    Cross-check the displacement between two captures: the cameras and the
    stage must have seen the SAME motion. Catches wrong-axis moves, sign
    flips, unit-factor mistakes and a dot that slipped on its mount.

    - magnitude: |Δcamera| vs |Δstage| (works from the 2nd capture on)
    - direction: angle between R@Δcamera and Δstage (needs the current fit R)

    Returns (message, ok).
    """
    dL = np.asarray(left, float) - np.asarray(prev_left, float)
    dZ = np.asarray(zaber, float) - np.asarray(prev_zaber, float)
    nL, nZ = float(np.linalg.norm(dL)), float(np.linalg.norm(dZ))

    if nZ < min_step and nL < min_step:
        return ("no motion since last capture — duplicate point?", False)
    if nZ < min_step:
        return (f"stage reports no motion but cameras saw Δ={nL:.3f} — "
                "wrong unit factor, or dot moved without the stage?", False)
    if nL < min_step:
        return (f"stage moved Δ={nZ:.3f} but cameras saw none — "
                "dot not rigidly mounted on the stage?", False)

    problems = []
    ratio = nL / nZ
    if abs(ratio - 1.0) > mag_tol:
        problems.append(f"step-size mismatch: camera Δ={nL:.3f} vs stage Δ={nZ:.3f} "
                        f"(ratio {ratio:.2f}) — check unit factor / mounting")
    ang = None
    if R is not None:
        v = np.asarray(R, float) @ dL
        denom = float(np.linalg.norm(v)) * nZ
        if denom > 1e-12:
            cosang = float(v @ dZ) / denom
            ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            if ang > ang_tol_deg:
                problems.append(f"direction mismatch: {ang:.0f}° between camera and stage motion "
                                "— wrong axis moved, or axis swapped/sign flipped?")
    if problems:
        return ("; ".join(problems), False)
    msg = f"motion OK: stage Δ=({dZ[0]:+.3f}, {dZ[1]:+.3f}, {dZ[2]:+.3f}), camera Δ={nL:.3f}"
    if ang is not None:
        msg += f", angle {ang:.1f}°"
    return (msg, True)


def geometry_hints(B: np.ndarray) -> list[str]:
    """Advice about the spatial distribution of the Zaber-side points."""
    hints = []
    n = len(B)
    if n < 3:
        return [f"{n} point(s) — need at least 3, aim for 8–15."]
    if n < 8:
        hints.append(f"only {n} points — aim for 8–15 spread over the working volume.")

    spans = B.max(axis=0) - B.min(axis=0)
    for axis, span in zip("xyz", spans):
        if span < 2.0:
            hints.append(f"{axis}-span only {span:.2f} — spread points over more of the travel range.")

    centered = B - B.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    if s[0] > 0:
        if s[1] / s[0] < 0.05:
            hints.append("points are nearly collinear — the fit is degenerate.")
        elif s[2] / s[0] < 0.05:
            hints.append("points are nearly coplanar — vary the third axis; rotation is ill-constrained.")

    if not hints:
        hints.append(f"geometry OK ({n} points, spans "
                     f"{spans[0]:.1f}/{spans[1]:.1f}/{spans[2]:.1f}).")
    return hints


def numpy_to_qpixmap_rgb(arr_rgb: np.ndarray) -> QPixmap:
    if arr_rgb.dtype != np.uint8:
        arr_rgb = arr_rgb.astype(np.uint8, copy=False)
    if not arr_rgb.flags.c_contiguous:
        arr_rgb = np.ascontiguousarray(arr_rgb)
    h, w, c = arr_rgb.shape
    qimg = QImage(arr_rgb.data, w, h, int(arr_rgb.strides[0]), QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def numpy_to_qpixmap_gray(arr: np.ndarray) -> QPixmap:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)


# ---------------- main GUI ----------------
class PointCaptureGUI(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self, use_dummy=False):
        super().__init__()
        self.setWindowTitle("LEFT → Zaber Point Capture")

        self.stereo = stereo_cameras
        self._detector = make_blob_detector()

        # capture data model: list of dicts
        # {"label": str, "left": [x,y,z]|None, "zaber": [x,y,z]|None,
        #  "tri_rms": float|None, "use": bool}
        self.rows: list[dict] = []
        self.fit_result: dict | None = None
        self._updating_table = False

        # live detection state
        self._last_left = None
        self._last_right = None
        self._last_uv = (None, None)          # (uvL, uvR)
        self._last_X_left = None
        self._last_tri_rms = None
        self._samples = deque(maxlen=200)     # (t, uvL, uvR) with both detected
        self._detect_counter = 0

        self.zaber = None                     # optional ZaberHumanInterface

        self._build_ui()

        # camera proxy (GUI stays usable without cameras, e.g. to refit a session)
        self.proxy = None
        try:
            self.proxy = DualCameraProxy(dtype="uint8", slots=8, use_dummy=use_dummy)
            self.proxy.start()
        except Exception as e:
            self.status_changed.emit(f"Camera proxy failed ({e}) — capture disabled, session/refit still works.")

        self._frame_q = queue.Queue(maxsize=2)
        self._reader_thread = None
        self._running = False

        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.on_tick)

        QShortcut(QKeySequence("Space"), self, activated=self.on_capture)

        if SESSION_PATH.exists():
            self.status_changed.emit(f"Previous session found ({SESSION_PATH.name}) — use 'Load Session' to restore.")

    # ---------------- UI ----------------
    def _build_ui(self):
        self.left_label = QLabel("Left")
        self.right_label = QLabel("Right")
        for lbl in (self.left_label, self.right_label):
            lbl.setFixedSize(560, 420)
            lbl.setStyleSheet("background-color: black;")
            lbl.setAlignment(Qt.AlignCenter)
            blank = QPixmap(lbl.size())
            blank.fill(Qt.black)
            lbl.setPixmap(blank)

        top = QHBoxLayout()
        top.addWidget(self.left_label)
        top.addWidget(self.right_label)

        # camera controls
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.left_cfg_btn = QPushButton("Left Config")
        self.right_cfg_btn = QPushButton("Right Config")
        self.overlay_checkbox = QCheckBox("Overlay dot")
        self.overlay_checkbox.setChecked(True)
        btns = QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.overlay_checkbox)
        btns.addStretch()
        btns.addWidget(self.left_cfg_btn)
        btns.addWidget(self.right_cfg_btn)

        # Zaber group
        zbox = QGroupBox("Zaber stage")
        zrow = QHBoxLayout(zbox)
        zrow.addWidget(QLabel("Port:"))
        self.zaber_port = QLineEdit("COM6")
        self.zaber_port.setFixedWidth(70)
        zrow.addWidget(self.zaber_port)
        self.zaber_connect_btn = QPushButton("Connect")
        zrow.addWidget(self.zaber_connect_btn)
        self.zaber_read_btn = QPushButton("Read Now")
        self.zaber_read_btn.setEnabled(False)
        zrow.addWidget(self.zaber_read_btn)
        self.zaber_auto = QCheckBox("Auto-read on capture")
        self.zaber_auto.setChecked(True)
        self.zaber_auto.setEnabled(False)
        zrow.addWidget(self.zaber_auto)
        zrow.addWidget(QLabel("Unit factor (µm→):"))
        self.unit_factor = QDoubleSpinBox()
        self.unit_factor.setDecimals(6)
        self.unit_factor.setRange(1e-9, 1e9)
        self.unit_factor.setValue(0.001)  # µm -> mm (LEFT frame is mm)
        self.unit_factor.setToolTip("Raw Zaber position (µm) is multiplied by this before storing.\n"
                                    "0.001 stores mm — matching the LEFT-camera frame units.")
        zrow.addWidget(self.unit_factor)
        zrow.addStretch()

        # manual coordinates + capture
        crow = QHBoxLayout()
        crow.addWidget(QLabel("X:"))
        self.x_input = QLineEdit(); self.x_input.setFixedWidth(90); self.x_input.setPlaceholderText("x")
        crow.addWidget(self.x_input)
        crow.addWidget(QLabel("Y:"))
        self.y_input = QLineEdit(); self.y_input.setFixedWidth(90); self.y_input.setPlaceholderText("y")
        crow.addWidget(self.y_input)
        crow.addWidget(QLabel("Z:"))
        self.z_input = QLineEdit(); self.z_input.setFixedWidth(90); self.z_input.setPlaceholderText("z")
        crow.addWidget(self.z_input)
        crow.addWidget(QLabel("Label:"))
        self.label_input = QLineEdit(); self.label_input.setFixedWidth(120); self.label_input.setPlaceholderText("optional")
        crow.addWidget(self.label_input)
        self.capture_btn = QPushButton("Capture (Space)")
        crow.addWidget(self.capture_btn)
        crow.addStretch()
        self.save_status = QLabel("Ready")
        crow.addWidget(self.save_status)

        # table
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            ["Use", "Label", "L x", "L y", "L z", "Z x", "Z y", "Z z", "tri px", "resid"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemChanged.connect(self._on_table_item_changed)

        # table buttons
        trow = QHBoxLayout()
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_all_btn = QPushButton("Remove All")
        self.load_session_btn = QPushButton("Load Session")
        trow.addWidget(self.remove_selected_btn)
        trow.addWidget(self.remove_all_btn)
        trow.addWidget(self.load_session_btn)
        trow.addStretch()
        self.save_as_btn = QPushButton("Save Transform As…")
        self.install_btn = QPushButton("Install as active (left_to_zaber.json)")
        trow.addWidget(self.save_as_btn)
        trow.addWidget(self.install_btn)

        # fit feedback
        self.fit_label = QLabel("Fit: —")
        self.fit_label.setStyleSheet("font-weight: bold;")
        self.motion_label = QLabel("Motion check: — (compares camera vs stage displacement between captures)")
        self.motion_label.setWordWrap(True)
        self.hints_label = QLabel("")
        self.hints_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(btns)
        layout.addWidget(zbox)
        layout.addLayout(crow)
        layout.addWidget(self.table)
        layout.addLayout(trow)
        layout.addWidget(self.fit_label)
        layout.addWidget(self.motion_label)
        layout.addWidget(self.hints_label)
        self.setLayout(layout)

        # wiring
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.left_cfg_btn.clicked.connect(lambda: self._open_config("left"))
        self.right_cfg_btn.clicked.connect(lambda: self._open_config("right"))
        self.capture_btn.clicked.connect(self.on_capture)
        self.remove_selected_btn.clicked.connect(self.on_remove_selected)
        self.remove_all_btn.clicked.connect(self.on_remove_all)
        self.load_session_btn.clicked.connect(self.on_load_session)
        self.save_as_btn.clicked.connect(self.on_save_as)
        self.install_btn.clicked.connect(self.on_install_active)
        self.zaber_connect_btn.clicked.connect(self.on_zaber_connect)
        self.zaber_read_btn.clicked.connect(self.on_zaber_read)
        self.status_changed.connect(self.save_status.setText)

    # ---------------- camera stream ----------------
    def _reader_loop(self):
        while self._running:
            try:
                left, right, ts = self.proxy.get_frames()
                if not self._frame_q.empty():
                    try:
                        self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_q.put((left, right, ts))
            except Exception as e:
                print(f"[PointCapture] Reader error: {e}")
                break

    def on_start(self):
        if self._running or self.proxy is None:
            if self.proxy is None:
                self.status_changed.emit("No camera proxy — restart the GUI with cameras attached.")
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
            left, right, _ = self._frame_q.get_nowait()
        except queue.Empty:
            return
        self._last_left = left
        self._last_right = right

        # detect every 2nd frame (blob detection is cheap, but keep the UI fluid)
        self._detect_counter = (self._detect_counter + 1) % 2
        if self._detect_counter == 0:
            self._update_detection(left, right)

        self._paint(left, right)

    def _update_detection(self, left_gray, right_gray):
        nearL, nearR = self._last_uv
        uvL = detect_dot(left_gray, detector=self._detector, near=nearL)
        uvR = detect_dot(right_gray, detector=self._detector, near=nearR)
        self._last_uv = (uvL, uvR)

        self._last_X_left = None
        self._last_tri_rms = None
        if uvL is not None and uvR is not None:
            try:
                X, rms = self.stereo.triangulate_best(uvL, uvR, refine=True)
                self._last_X_left = np.asarray(X, float).reshape(3)
                self._last_tri_rms = float(rms)
                self._samples.append((time.monotonic(), uvL, uvR))
            except Exception:
                pass

    def _paint(self, left_gray, right_gray):
        if not self.overlay_checkbox.isChecked():
            self._set_pixmap_fit(self.left_label, numpy_to_qpixmap_gray(left_gray))
            self._set_pixmap_fit(self.right_label, numpy_to_qpixmap_gray(right_gray))
            return

        uvL, uvR = self._last_uv
        for gray, uv, label in ((left_gray, uvL, self.left_label), (right_gray, uvR, self.right_label)):
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            if uv is not None:
                u, v = int(round(uv[0])), int(round(uv[1]))
                cv2.circle(vis, (u, v), 6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawMarker(vis, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                cv2.putText(vis, f"({uv[0]:.1f}, {uv[1]:.1f})", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            if self._last_X_left is not None:
                X = self._last_X_left
                rms = self._last_tri_rms
                ok = rms is not None and rms <= TRI_RMS_WARN_PX
                color = (0, 255, 0) if ok else (255, 80, 0)
                cv2.putText(vis, f"X=({X[0]:.2f}, {X[1]:.2f}, {X[2]:.2f}) mm  rms={rms:.2f}px",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                if not ok:
                    cv2.putText(vis, "CHECK DETECTION (L/R mismatch?)", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 0), 2, cv2.LINE_AA)
            self._set_pixmap_fit(label, numpy_to_qpixmap_rgb(vis))

    def _set_pixmap_fit(self, label: QLabel, pm: QPixmap):
        if pm.width() == label.width() and pm.height() == label.height():
            label.setPixmap(pm)
            return
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def _open_config(self, side: str):
        dlg = AlliedConfigDialog(side, lambda cfg: self._apply_config(side, cfg), parent=self)
        dlg.exec_()

    def _apply_config(self, side: str, cfg_obj):
        was_running = self._running
        if was_running:
            self.on_stop()
        try:
            if side == "left":
                self.proxy.set_configs(cfg_left=cfg_obj, cfg_right=None)
            else:
                self.proxy.set_configs(cfg_left=None, cfg_right=cfg_obj)
        finally:
            if was_running:
                self.on_start()

    # ---------------- Zaber ----------------
    def on_zaber_connect(self):
        if self.zaber is not None:
            try:
                self.zaber.close()
            except Exception:
                pass
            self.zaber = None
            self.zaber_connect_btn.setText("Connect")
            self.zaber_read_btn.setEnabled(False)
            self.zaber_auto.setEnabled(False)
            self.status_changed.emit("Zaber disconnected")
            return
        try:
            from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import (
                ZaberHumanInterface,
            )
            # home_on_connect=False: attach WITHOUT homing/moving the stage —
            # a connect during calibration must never cause physical motion.
            self.zaber = ZaberHumanInterface(port=self.zaber_port.text().strip(), home_on_connect=False)
            self.zaber_connect_btn.setText("Disconnect")
            self.zaber_read_btn.setEnabled(True)
            self.zaber_auto.setEnabled(True)
            self.status_changed.emit(f"Zaber connected on {self.zaber_port.text().strip()} (no homing)")
        except Exception as e:
            self.zaber = None
            self.status_changed.emit(f"Zaber connect failed: {e}")
            QMessageBox.warning(self, "Zaber", f"Could not connect:\n{e}\n\n"
                                "Note: the port is busy if the Human Interface GUI is running.")

    def _read_zaber_scaled(self):
        pos = self.zaber.get_position()  # µm
        f = float(self.unit_factor.value())
        return [float(pos[0]) * f, float(pos[1]) * f, float(pos[2]) * f]

    def on_zaber_read(self):
        if self.zaber is None:
            return
        try:
            x, y, z = self._read_zaber_scaled()
            self.x_input.setText(f"{x:.4f}")
            self.y_input.setText(f"{y:.4f}")
            self.z_input.setText(f"{z:.4f}")
            self.status_changed.emit(f"Zaber: ({x:.4f}, {y:.4f}, {z:.4f})")
        except Exception as e:
            self.status_changed.emit(f"Zaber read failed: {e}")

    # ---------------- capture ----------------
    def _read_xyz_inputs(self):
        def to_float(s):
            s = (s or "").strip()
            try:
                return float(s)
            except ValueError:
                return None
        vals = (to_float(self.x_input.text()), to_float(self.y_input.text()), to_float(self.z_input.text()))
        return list(vals) if all(v is not None for v in vals) else None

    def on_capture(self):
        # 1) averaged stereo point from the recent sample window
        now = time.monotonic()
        recent = [(uL, uR) for (t, uL, uR) in self._samples if now - t <= CAPTURE_WINDOW_S]
        left_xyz = None
        tri_rms = None
        if len(recent) >= CAPTURE_MIN_SAMPLES:
            arrL = np.array([r[0] for r in recent], float)
            arrR = np.array([r[1] for r in recent], float)
            uvL = tuple(np.median(arrL, axis=0))
            uvR = tuple(np.median(arrR, axis=0))
            jitter = float(max(arrL.std(axis=0).max(), arrR.std(axis=0).max()))
            try:
                X, rms = self.stereo.triangulate_best(uvL, uvR, refine=True)
                left_xyz = [float(X[0]), float(X[1]), float(X[2])]
                tri_rms = float(rms)
                if rms > TRI_RMS_WARN_PX:
                    self.status_changed.emit(f"Warning: triangulation rms {rms:.2f}px — L/R may see different blobs.")
                elif jitter > 1.0:
                    self.status_changed.emit(f"Note: detection jitter {jitter:.2f}px over the last {CAPTURE_WINDOW_S:.0f}s.")
            except Exception as e:
                self.status_changed.emit(f"Triangulation failed: {e}")
        elif self._last_X_left is not None:
            left_xyz = [float(v) for v in self._last_X_left]
            tri_rms = self._last_tri_rms
            self.status_changed.emit("Only a single-frame detection was available (hold the target still ~2s for averaging).")
        else:
            self.status_changed.emit("No dot detected in both cameras — nothing captured.")
            return

        # 2) Zaber coordinates: auto-read or manual fields
        zaber_xyz = None
        if self.zaber is not None and self.zaber_auto.isChecked():
            try:
                zaber_xyz = self._read_zaber_scaled()
                self.x_input.setText(f"{zaber_xyz[0]:.4f}")
                self.y_input.setText(f"{zaber_xyz[1]:.4f}")
                self.z_input.setText(f"{zaber_xyz[2]:.4f}")
            except Exception as e:
                self.status_changed.emit(f"Zaber auto-read failed: {e}")
        if zaber_xyz is None:
            zaber_xyz = self._read_xyz_inputs()
        if zaber_xyz is None:
            self.status_changed.emit("Zaber coordinates missing — connect the stage or fill X/Y/Z.")

        new_row = {
            "label": self.label_input.text().strip(),
            "left": left_xyz,
            "zaber": zaber_xyz,
            "tri_rms": tri_rms,
            "use": True,
        }

        # 3) motion consistency vs the previous complete capture:
        #    the stage and the cameras must have seen the same displacement.
        prev = next((r for r in reversed(self.rows) if self._row_valid(r) and r["use"]), None)
        if prev is not None and self._row_valid(new_row):
            R = self.fit_result["T"].R if self.fit_result is not None else None
            msg, ok = motion_check(prev["left"], prev["zaber"],
                                   new_row["left"], new_row["zaber"], R=R)
            self.motion_label.setText(f"Motion check: {msg}")
            self.motion_label.setStyleSheet("color: #007700;" if ok else "color: #cc6600; font-weight: bold;")

        self.rows.append(new_row)
        self._refresh_table()
        self._refit()
        self._autosave_session()
        self.status_changed.emit(f"Captured point #{len(self.rows) - 1:03d}")

    # ---------------- table ----------------
    def _refresh_table(self):
        self._updating_table = True
        try:
            residuals = {}
            if self.fit_result is not None:
                used = [i for i, r in enumerate(self.rows) if self._row_valid(r) and r["use"]]
                for i, res in zip(used, self.fit_result["residuals"]):
                    residuals[i] = float(res)
            worst = max(residuals, key=residuals.get) if len(residuals) >= 3 else None

            self.table.setRowCount(len(self.rows))
            for i, r in enumerate(self.rows):
                chk = QTableWidgetItem()
                chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                chk.setCheckState(Qt.Checked if r["use"] else Qt.Unchecked)
                self.table.setItem(i, 0, chk)

                def cell(text, red=False):
                    it = QTableWidgetItem(text)
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    if red:
                        it.setForeground(QColor("#cc0000"))
                    return it

                self.table.setItem(i, 1, cell(r["label"] or ""))
                for k in range(3):
                    self.table.setItem(i, 2 + k, cell(f"{r['left'][k]:.3f}" if r["left"] else "–"))
                for k in range(3):
                    self.table.setItem(i, 5 + k, cell(f"{r['zaber'][k]:.3f}" if r["zaber"] else "–"))
                bad_tri = r["tri_rms"] is not None and r["tri_rms"] > TRI_RMS_WARN_PX
                self.table.setItem(i, 8, cell(f"{r['tri_rms']:.2f}" if r["tri_rms"] is not None else "–", red=bad_tri))
                res = residuals.get(i)
                self.table.setItem(i, 9, cell(f"{res:.4f}" if res is not None else "–", red=(i == worst)))
        finally:
            self._updating_table = False

    def _on_table_item_changed(self, item):
        if self._updating_table or item.column() != 0:
            return
        i = item.row()
        if 0 <= i < len(self.rows):
            self.rows[i]["use"] = item.checkState() == Qt.Checked
            self._refit()
            self._refresh_table()
            self._autosave_session()

    def on_remove_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            self.status_changed.emit("No row selected")
            return
        for i in rows:
            if 0 <= i < len(self.rows):
                del self.rows[i]
        self._refit()
        self._refresh_table()
        self._autosave_session()
        self.status_changed.emit(f"Removed {len(rows)} row(s)")

    def on_remove_all(self):
        if self.rows and QMessageBox.question(self, "Remove all", f"Discard all {len(self.rows)} points?") != QMessageBox.Yes:
            return
        self.rows.clear()
        self._refit()
        self._refresh_table()
        self._autosave_session()

    # ---------------- fitting ----------------
    @staticmethod
    def _row_valid(r) -> bool:
        return (r["left"] is not None and r["zaber"] is not None
                and len(r["left"]) == 3 and len(r["zaber"]) == 3)

    def _used_AB(self):
        rows = [r for r in self.rows if r["use"] and self._row_valid(r)]
        if len(rows) < 3:
            return None, None
        A = np.array([r["left"] for r in rows], float)
        B = np.array([r["zaber"] for r in rows], float)
        return A, B

    def _refit(self):
        self.fit_result = None
        A, B = self._used_AB()
        if A is None:
            self.fit_label.setText(f"Fit: need ≥3 complete included points "
                                   f"(have {sum(1 for r in self.rows if r['use'] and self._row_valid(r))}).")
            self.hints_label.setText("")
            return
        try:
            self.fit_result = fit_pairs(A, B)
        except Exception as e:
            self.fit_label.setText(f"Fit failed: {e}")
            self.hints_label.setText("")
            return

        rms = self.fit_result["rms"]
        scale = self.fit_result["scale"]
        txt = f"Fit: N={len(A)}  RMS={rms:.4f} (zaber units)  |  scale check: {scale:.4f}"
        style = "font-weight: bold;"
        if np.isfinite(scale) and abs(scale - 1.0) > SCALE_WARN:
            txt += "  ⚠ scale far from 1 — LEFT (mm) and Zaber values use different units?"
            style += " color: #cc6600;"
        self.fit_label.setStyleSheet(style)
        self.fit_label.setText(txt)

        hints = geometry_hints(B)
        self.hints_label.setText("Hints: " + " | ".join(hints))

    # ---------------- session persistence ----------------
    def _autosave_session(self):
        try:
            with open(SESSION_PATH, "w", encoding="utf-8") as f:
                json.dump({"rows": self.rows, "saved_at": datetime.datetime.now().isoformat()}, f, indent=2)
        except Exception as e:
            print(f"[PointCapture] session autosave failed: {e}")

    def on_load_session(self):
        path = SESSION_PATH
        if not path.exists():
            p, _ = QFileDialog.getOpenFileName(self, "Load Session JSON", "", "JSON (*.json)")
            if not p:
                return
            path = Path(p)
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.rows = d["rows"]
            self._refit()
            self._refresh_table()
            self.status_changed.emit(f"Loaded {len(self.rows)} points from {path.name}")
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    # ---------------- saving the transform ----------------
    def _build_transform_json(self) -> dict | None:
        if self.fit_result is None:
            QMessageBox.information(self, "No fit", "Need a valid fit first (≥3 complete included points).")
            return None
        A, B = self._used_AB()
        T: SE3 = self.fit_result["T"]
        return {
            "name": "left_to_zaber",
            "description": "Transformation from left camera frame to zaber coordinate system",
            "R": T.R.tolist(),
            "t": T.t.tolist(),
            "source": {
                "method": "umeyama_fit",
                "num_points": int(len(A)),
                "rms_error": self.fit_result["rms"],
                "scale_check": self.fit_result["scale"],
                "unit_factor_um": float(self.unit_factor.value()),
            },
            "data": {
                "all_pairs": [{"left": r["left"], "input": r["zaber"], "use": r["use"],
                               "label": r["label"], "tri_rms": r["tri_rms"]} for r in self.rows],
                "valid_pairs": {"left": A.tolist(), "input": B.tolist()},
            },
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

    def on_save_as(self):
        obj = self._build_transform_json()
        if obj is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Transform JSON", "left_to_zaber.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
            self.status_changed.emit(f"Saved transform to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Write error", str(e))

    def on_install_active(self):
        obj = self._build_transform_json()
        if obj is None:
            return
        msg = (f"Overwrite the ACTIVE transform?\n\n{ACTIVE_TRANSFORM_PATH}\n\n"
               f"N={obj['source']['num_points']}, RMS={obj['source']['rms_error']:.4f}\n"
               f"The current file will be backed up first.")
        if QMessageBox.question(self, "Install transform", msg) != QMessageBox.Yes:
            return
        try:
            if ACTIVE_TRANSFORM_PATH.exists():
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = ACTIVE_TRANSFORM_PATH.with_name(f"left_to_zaber_backup_{stamp}.json")
                shutil.copy2(ACTIVE_TRANSFORM_PATH, backup)
                self.status_changed.emit(f"Backed up old transform to {backup.name}")
            with open(ACTIVE_TRANSFORM_PATH, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
            QMessageBox.information(self, "Installed",
                                    f"Active transform updated:\n{ACTIVE_TRANSFORM_PATH}\n\n"
                                    "Restart the Human Interface GUI to load it.")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Install failed", str(e))

    # ---------------- shutdown ----------------
    def closeEvent(self, ev):
        try:
            self.on_stop()
        finally:
            if self.zaber is not None:
                try:
                    self.zaber.close()
                except Exception:
                    pass
            if self.proxy is not None:
                try:
                    self.proxy.shutdown()
                except Exception:
                    pass
        ev.accept()


# Backward-compatible alias (old name)
DualCamImageCapture = PointCaptureGUI


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # Windows

    app = QApplication(sys.argv)
    w = PointCaptureGUI(use_dummy=False)  # set True to run without hardware
    w.resize(1180, 980)
    w.show()
    sys.exit(app.exec_())
