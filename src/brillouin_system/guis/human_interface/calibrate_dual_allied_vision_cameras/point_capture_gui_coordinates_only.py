"""
Point-capture GUI for the LEFT-camera -> Zaber (rig) coordinate transform.

Rig-frame convention (must match the runtime, see hi_frontend/eye_tracker):
  - The frame MOVES WITH the zaber stage (cameras + laser ride on it).
  - x = y = 0 : on the laser beam axis.
  - z = 0    : laser focus with the eye lens at 0 µm.
  - The dot is FIXED in the room, so its rig coordinates are
        dot_rig = -(stage - reference_pose)        [mm]
    where the reference pose is the stage position at which the laser
    (lens parked at 0 µm) is centered and focused ON the dot.

Workflow:
  1. Start cameras. Connect stage (COM6) and lens (COM5) — both attach
     WITHOUT homing. Move the lens to 0 µm (button) and leave it there.
  2. Jog the stage until the laser hits the dot and is focused on it,
     then press "Set Reference Pose".
  3. Jog the stage to a new pose (the dot moves opposite in the images),
     hold still ~2 s, press Capture (Space). Rig coordinates are computed
     automatically; images are saved if a folder is chosen.
  4. The fit updates after every capture: RMS, per-point residuals,
     scale check, motion check, geometry hints, live 3D view.
  5. "Install as active" writes stereo_configs/left_to_zaber.json
     (with a timestamped backup). Restart the HI GUI afterwards.
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
    QFileDialog, QLineEdit, QShortcut, QCheckBox,
    QTableWidget, QTableWidgetItem, QGroupBox, QMessageBox, QHeaderView,
    QGridLayout,
)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QColor
from PyQt5.QtCore import QTimer, pyqtSignal, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

UM_TO_MM = 1e-3
TRI_RMS_WARN_PX = 2.0        # triangulation reprojection RMS above this = suspicious detection
SCALE_WARN = 0.02            # |scale-1| above this suggests a units mismatch
CAPTURE_WINDOW_S = 2.0       # average detections over this window on capture
CAPTURE_MIN_SAMPLES = 5
LENS_MOVED_TOL_UM = 10.0     # lens must not move after the reference pose is set

FRAME_INFO = (
    "Rig frame: x=y=0 on the laser axis; z anchored to the lens scale (z=0 = focus @ lens 0 µm). "
    "Park the lens at its working position (e.g. 10 mm init), FOCUS the laser on the dot, Set Reference Pose "
    "— the dot is then at (0, 0, lens reading in mm). Do not move the lens afterwards. "
    "Captured dot coords: x,y = −Δstage, z = lens_ref − Δstage_z [mm]."
)


# ---------------- pure helpers (testable without Qt) ----------------
def rig_coords_from_stage(stage_um, reference_um, lens_ref_um: float = 0.0) -> list[float]:
    """
    Dot position in the moving rig frame [mm].

    x,y,z all move as -(stage - reference); the absolute z level is anchored
    to the lens scale: at the reference pose (laser focused on the dot) the
    dot sits at z = lens reading in mm (runtime convention: focus z = lens mm).
    """
    s = np.asarray(stage_um, float)
    r = np.asarray(reference_um, float)
    out = -(s - r) * UM_TO_MM
    out[2] += float(lens_ref_um) * UM_TO_MM
    return list(out)


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
    msg = f"motion OK: rig Δ=({dZ[0]:+.3f}, {dZ[1]:+.3f}, {dZ[2]:+.3f}), camera Δ={nL:.3f}"
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


# ---------------- 3D viewer (live) ----------------
class Fit3DDialog(QWidget):
    """Live 3D view: zaber points, fitted LEFT points mapped into the zaber
    frame, residual lines, rig axes at the origin, LEFT-camera axes at its
    fitted pose."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Fitted Coordinate System & Points")
        self.fig = Figure(figsize=(7, 6), constrained_layout=True)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.fig)
        self.info = QLabel("—")
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.info)
        self.setLayout(layout)
        self.resize(820, 720)

    def _plot_axes(self, origin, R, length, prefix):
        o = np.asarray(origin, float).reshape(3)
        colors = ("r", "g", "b")
        for k, (axis_name, c) in enumerate(zip("xyz", colors)):
            d = np.asarray(R, float)[:, k]
            seg = np.vstack([o, o + length * d])
            self.ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=c, linewidth=2,
                         label=f"{prefix}{axis_name}")

    def _set_equal(self, X):
        if X is None or X.size == 0:
            return
        mins, maxs = X.min(axis=0), X.max(axis=0)
        center = (mins + maxs) / 2
        half = max(float(np.max(maxs - mins)) / 2, 1.0)
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        self.ax.set_box_aspect((1, 1, 1))

    def update_data(self, A: np.ndarray | None, B: np.ndarray | None,
                    T: SE3 | None, rms: float | None = None):
        self.ax.clear()
        self.ax.set_xlabel("rig X [mm]")
        self.ax.set_ylabel("rig Y [mm]")
        self.ax.set_zlabel("rig Z [mm]")
        self.ax.set_title("Rig frame (z=0: laser focus @ lens 0)")

        pts = []
        # rig frame at origin
        length = 5.0
        self._plot_axes(np.zeros(3), np.eye(3), length, "rig-")
        self.ax.scatter([0], [0], [0], color="k", s=30)

        if B is not None and len(B):
            B = np.asarray(B, float)
            self.ax.scatter(B[:, 0], B[:, 1], B[:, 2], marker="^", s=40,
                            color="tab:orange", label="stage points (rig)")
            pts.append(B)

        if T is not None and A is not None and len(A):
            A = np.asarray(A, float)
            TA = T.apply_points(A)
            self.ax.scatter(TA[:, 0], TA[:, 1], TA[:, 2], marker="o", s=25,
                            color="tab:blue", label="camera points → rig (fit)")
            pts.append(TA)
            if B is not None and len(B) == len(TA):
                for p, q in zip(TA, B):
                    seg = np.vstack([p, q])
                    self.ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                                 color="gray", linewidth=0.8, alpha=0.7)
            # LEFT camera pose in the rig frame: origin = T.t, axes = columns of T.R
            self._plot_axes(T.t, T.R, length, "cam-")
            pts.append(T.t.reshape(1, 3))

        if pts:
            self._set_equal(np.vstack(pts))
        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()
        if rms is not None:
            self.info.setText(f"N = {0 if A is None else len(A)}   fit RMS = {rms:.4f} mm")
        else:
            self.info.setText("No fit yet — capture ≥3 points.")


# ---------------- main GUI ----------------
class PointCaptureGUI(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self, use_dummy=False):
        super().__init__()
        self.setWindowTitle("LEFT → Zaber Point Capture")

        self.stereo = stereo_cameras
        self._detector = make_blob_detector()  # dark dot on bright paper

        # capture data model: list of dicts
        # {"label", "left":[x,y,z]|None, "zaber":[x,y,z]|None, "stage_um":[..]|None,
        #  "tri_rms": float|None, "use": bool, "base": str|None}
        self.rows: list[dict] = []
        self.fit_result: dict | None = None
        self._updating_table = False

        # live detection state
        self._last_left = None
        self._last_right = None
        self._last_uv = (None, None)
        self._last_X_left = None
        self._last_tri_rms = None
        self._samples = deque(maxlen=200)
        self._detect_counter = 0

        # devices
        self.stage = None                     # ZaberHumanInterface (COM6)
        self.lens = None                      # ZaberEyeLens (COM5)
        self.reference_um: list[float] | None = None
        self.reference_lens_um: float = 0.0   # lens reading when the reference pose was set

        # transform overlay
        self.loaded_T: SE3 | None = None
        self.loaded_T_name: str = ""

        # image saving
        self.save_dir: Path | None = None

        self.viewer3d: Fit3DDialog | None = None

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
            lbl.setFixedSize(520, 390)
            lbl.setStyleSheet("background-color: black;")
            lbl.setAlignment(Qt.AlignCenter)
            blank = QPixmap(lbl.size())
            blank.fill(Qt.black)
            lbl.setPixmap(blank)

        top = QHBoxLayout()
        top.addWidget(self.left_label)
        top.addWidget(self.right_label)

        # frame-convention banner
        self.banner = QLabel(FRAME_INFO)
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("background: #fff6d6; border: 1px solid #d8c775; padding: 4px;")

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

        # ---- Stage group (frontend-style jog) ----
        sbox = QGroupBox("Zaber stage (COM6) — rig moves, dot appears opposite")
        sgrid = QGridLayout(sbox)
        srow0 = QHBoxLayout()
        self.stage_port = QLineEdit("COM6"); self.stage_port.setFixedWidth(60)
        self.stage_connect_btn = QPushButton("Connect")
        srow0.addWidget(QLabel("Port:")); srow0.addWidget(self.stage_port)
        srow0.addWidget(self.stage_connect_btn); srow0.addStretch()
        sgrid.addLayout(srow0, 0, 0, 1, 5)

        self.stage_pos_labels = {}
        self.stage_step_inputs = {}
        for i, ax in enumerate("xyz"):
            pos_lbl = QLabel(f"{ax.upper()} — µm")
            step = QLineEdit("1000"); step.setFixedWidth(60)
            minus = QPushButton(f"−{ax}"); plus = QPushButton(f"+{ax}")
            minus.setFixedWidth(45); plus.setFixedWidth(45)
            minus.clicked.connect(lambda _, a=ax: self._jog_stage(a, -1))
            plus.clicked.connect(lambda _, a=ax: self._jog_stage(a, +1))
            sgrid.addWidget(pos_lbl, i + 1, 0)
            sgrid.addWidget(QLabel("step µm:"), i + 1, 1)
            sgrid.addWidget(step, i + 1, 2)
            sgrid.addWidget(minus, i + 1, 3)
            sgrid.addWidget(plus, i + 1, 4)
            self.stage_pos_labels[ax] = pos_lbl
            self.stage_step_inputs[ax] = step

        self.set_ref_btn = QPushButton("Set Reference Pose (laser focused on dot)")
        self.set_ref_btn.setEnabled(False)
        self.ref_label = QLabel("Reference: not set")
        sgrid.addWidget(self.set_ref_btn, 4, 0, 1, 3)
        sgrid.addWidget(self.ref_label, 4, 3, 1, 2)

        # ---- Lens group ----
        lbox = QGroupBox("Eye lens (COM5) — park at working position, do NOT move after Set Reference")
        lrow = QHBoxLayout(lbox)
        self.lens_port = QLineEdit("COM5"); self.lens_port.setFixedWidth(60)
        self.lens_connect_btn = QPushButton("Connect")
        self.lens_target_input = QLineEdit("12000"); self.lens_target_input.setFixedWidth(70)
        self.lens_move_btn = QPushButton("Move Lens")
        self.lens_move_btn.setEnabled(False)
        self.lens_pos_label = QLabel("Lens: — µm")
        lrow.addWidget(QLabel("Port:")); lrow.addWidget(self.lens_port)
        lrow.addWidget(self.lens_connect_btn)
        lrow.addWidget(QLabel("target µm:")); lrow.addWidget(self.lens_target_input)
        lrow.addWidget(self.lens_move_btn)
        lrow.addWidget(self.lens_pos_label)
        lrow.addStretch()

        dev_row = QHBoxLayout()
        dev_row.addWidget(sbox, stretch=3)
        dev_row.addWidget(lbox, stretch=2)

        # ---- capture row ----
        crow = QHBoxLayout()
        self.auto_fill_checkbox = QCheckBox("Auto rig coords = −(stage − ref)")
        self.auto_fill_checkbox.setChecked(True)
        crow.addWidget(self.auto_fill_checkbox)
        crow.addWidget(QLabel("X:"))
        self.x_input = QLineEdit(); self.x_input.setFixedWidth(80); self.x_input.setPlaceholderText("mm")
        crow.addWidget(self.x_input)
        crow.addWidget(QLabel("Y:"))
        self.y_input = QLineEdit(); self.y_input.setFixedWidth(80); self.y_input.setPlaceholderText("mm")
        crow.addWidget(self.y_input)
        crow.addWidget(QLabel("Z:"))
        self.z_input = QLineEdit(); self.z_input.setFixedWidth(80); self.z_input.setPlaceholderText("mm")
        crow.addWidget(self.z_input)
        crow.addWidget(QLabel("Label:"))
        self.label_input = QLineEdit(); self.label_input.setFixedWidth(100); self.label_input.setPlaceholderText("optional")
        crow.addWidget(self.label_input)
        self.capture_btn = QPushButton("Capture (Space)")
        crow.addWidget(self.capture_btn)
        crow.addStretch()
        self.save_status = QLabel("Ready")
        crow.addWidget(self.save_status)

        # ---- image saving + transform overlay row ----
        orow = QHBoxLayout()
        self.folder_btn = QPushButton("Image Folder…")
        self.save_images_checkbox = QCheckBox("Save images on capture")
        self.save_images_checkbox.setChecked(True)
        self.folder_label = QLabel("no folder")
        orow.addWidget(self.folder_btn)
        orow.addWidget(self.save_images_checkbox)
        orow.addWidget(self.folder_label)
        orow.addStretch()
        self.track_fit_checkbox = QCheckBox("Overlay ZABER coords (current fit)")
        self.load_T_btn = QPushButton("Load Transform JSON…")
        self.transform_src_label = QLabel("")
        orow.addWidget(self.track_fit_checkbox)
        orow.addWidget(self.load_T_btn)
        orow.addWidget(self.transform_src_label)

        # ---- table ----
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            ["Use", "Label", "L x", "L y", "L z", "Rig x", "Rig y", "Rig z", "tri px", "resid"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemChanged.connect(self._on_table_item_changed)

        trow = QHBoxLayout()
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_all_btn = QPushButton("Remove All")
        self.load_session_btn = QPushButton("Load Session")
        self.view3d_btn = QPushButton("Show 3D View")
        trow.addWidget(self.remove_selected_btn)
        trow.addWidget(self.remove_all_btn)
        trow.addWidget(self.load_session_btn)
        trow.addWidget(self.view3d_btn)
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
        layout.addWidget(self.banner)
        layout.addLayout(btns)
        layout.addLayout(dev_row)
        layout.addLayout(crow)
        layout.addLayout(orow)
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
        self.view3d_btn.clicked.connect(self.on_show_3d)
        self.save_as_btn.clicked.connect(self.on_save_as)
        self.install_btn.clicked.connect(self.on_install_active)
        self.stage_connect_btn.clicked.connect(self.on_stage_connect)
        self.lens_connect_btn.clicked.connect(self.on_lens_connect)
        self.lens_move_btn.clicked.connect(self.on_lens_move)
        self.set_ref_btn.clicked.connect(self.on_set_reference)
        self.folder_btn.clicked.connect(self.on_choose_folder)
        self.load_T_btn.clicked.connect(self.on_load_transform)
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

    def _active_transform(self) -> tuple[SE3 | None, str]:
        if self.track_fit_checkbox.isChecked() and self.fit_result is not None:
            return self.fit_result["T"], "current fit"
        if self.loaded_T is not None:
            return self.loaded_T, self.loaded_T_name
        return None, ""

    def _paint(self, left_gray, right_gray):
        if not self.overlay_checkbox.isChecked():
            self._set_pixmap_fit(self.left_label, numpy_to_qpixmap_gray(left_gray))
            self._set_pixmap_fit(self.right_label, numpy_to_qpixmap_gray(right_gray))
            return

        T, T_name = self._active_transform()
        uvL, uvR = self._last_uv
        for gray, uv, label in ((left_gray, uvL, self.left_label), (right_gray, uvR, self.right_label)):
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            if uv is not None:
                u, v = int(round(uv[0])), int(round(uv[1]))
                cv2.circle(vis, (u, v), 6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawMarker(vis, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            if self._last_X_left is not None:
                X = self._last_X_left
                rms = self._last_tri_rms
                ok = rms is not None and rms <= TRI_RMS_WARN_PX
                color = (0, 255, 0) if ok else (255, 80, 0)
                cv2.putText(vis, f"LEFT=({X[0]:.2f}, {X[1]:.2f}, {X[2]:.2f}) mm  rms={rms:.2f}px",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
                if T is not None:
                    Xz = T.apply_points(X)
                    cv2.putText(vis, f"ZABER=({Xz[0]:.2f}, {Xz[1]:.2f}, {Xz[2]:.2f}) mm [{T_name}]",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
                if not ok:
                    cv2.putText(vis, "CHECK DETECTION (L/R mismatch?)", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2, cv2.LINE_AA)
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

    # ---------------- Zaber stage ----------------
    def on_stage_connect(self):
        if self.stage is not None:
            try:
                self.stage.close()
            except Exception:
                pass
            self.stage = None
            self.stage_connect_btn.setText("Connect")
            self.set_ref_btn.setEnabled(False)
            self.status_changed.emit("Stage disconnected")
            return
        try:
            from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import (
                ZaberHumanInterface,
            )
            # never home/move on connect during calibration
            self.stage = ZaberHumanInterface(port=self.stage_port.text().strip(), home_on_connect=False)
            self.stage_connect_btn.setText("Disconnect")
            self.set_ref_btn.setEnabled(True)
            self._update_stage_display()
            self.status_changed.emit(f"Stage connected on {self.stage_port.text().strip()} (no homing)")
        except Exception as e:
            self.stage = None
            self.status_changed.emit(f"Stage connect failed: {e}")
            QMessageBox.warning(self, "Zaber stage", f"Could not connect:\n{e}\n\n"
                                "Note: the port is busy if the Human Interface GUI is running.")

    def _jog_stage(self, axis: str, sign: int):
        if self.stage is None:
            self.status_changed.emit("Stage not connected")
            return
        try:
            step = float(self.stage_step_inputs[axis].text())
        except ValueError:
            self.status_changed.emit(f"Invalid step for {axis}")
            return
        try:
            kwargs = {f"d{axis}": sign * step}
            self.stage.move_rel(**kwargs)
            self._update_stage_display()
        except Exception as e:
            self.status_changed.emit(f"Stage move failed: {e}")

    def _update_stage_display(self):
        if self.stage is None:
            return
        try:
            x, y, z = self.stage.get_position()
            for ax, val in zip("xyz", (x, y, z)):
                self.stage_pos_labels[ax].setText(f"{ax.upper()} {val:.1f} µm")
        except Exception as e:
            self.status_changed.emit(f"Stage read failed: {e}")

    def on_set_reference(self):
        if self.stage is None:
            return
        try:
            self.reference_um = list(self.stage.get_position())
        except Exception as e:
            self.status_changed.emit(f"Stage read failed: {e}")
            return
        lens_note = " (lens not connected — assuming lens at 0 µm for the z anchor)"
        if self.lens is not None:
            try:
                self.reference_lens_um = float(self.lens.get_position())
                lens_note = f" (lens at {self.reference_lens_um:.1f} µm — keep it there)"
            except Exception:
                pass
        r = self.reference_um
        self.ref_label.setText(f"Reference: ({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}) µm | "
                               f"lens {self.reference_lens_um:.1f} µm")
        self.status_changed.emit(
            f"Reference set — dot is at rig (0, 0, {self.reference_lens_um * UM_TO_MM:.3f} mm)." + lens_note)
        self._autosave_session()

    # ---------------- Eye lens ----------------
    def on_lens_connect(self):
        if self.lens is not None:
            try:
                self.lens.close()
            except Exception:
                pass
            self.lens = None
            self.lens_connect_btn.setText("Connect")
            self.lens_move_btn.setEnabled(False)
            self.lens_pos_label.setText("Lens: — µm")
            self.status_changed.emit("Lens disconnected")
            return
        try:
            from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
            self.lens = ZaberEyeLens(port=self.lens_port.text().strip(), home_on_connect=False)
            self.lens_connect_btn.setText("Disconnect")
            self.lens_move_btn.setEnabled(True)
            self._update_lens_display()
            self.status_changed.emit(f"Lens connected on {self.lens_port.text().strip()} (no homing)")
        except Exception as e:
            self.lens = None
            self.status_changed.emit(f"Lens connect failed: {e}")
            QMessageBox.warning(self, "Eye lens", f"Could not connect:\n{e}")

    def on_lens_move(self):
        if self.lens is None:
            return
        try:
            target = float(self.lens_target_input.text())
        except ValueError:
            self.status_changed.emit("Invalid lens target (µm)")
            return
        if self.reference_um is not None and abs(target - self.reference_lens_um) > LENS_MOVED_TOL_UM:
            if QMessageBox.question(
                    self, "Lens move",
                    "A reference pose is already set — moving the lens breaks the z convention.\n"
                    "Move anyway? (You must then set a NEW reference pose.)") != QMessageBox.Yes:
                return
        try:
            self.lens.move_abs(target)
            self._update_lens_display()
            self.status_changed.emit(f"Lens parked at {target:.1f} µm — focus the laser on the dot, "
                                     "then Set Reference Pose, and leave the lens there.")
        except Exception as e:
            self.status_changed.emit(f"Lens move failed: {e}")

    def _update_lens_display(self):
        if self.lens is None:
            return
        try:
            pos = float(self.lens.get_position())
            txt = f"Lens: {pos:.1f} µm → focus z = {pos * UM_TO_MM:.3f} mm"
            if self.reference_um is not None:
                moved = abs(pos - self.reference_lens_um) > LENS_MOVED_TOL_UM
                txt += "  ⚠ MOVED since reference!" if moved else "  ✓"
                self.lens_pos_label.setStyleSheet(
                    "color: #cc0000; font-weight: bold;" if moved else "color: #007700;")
            else:
                self.lens_pos_label.setStyleSheet("")
            self.lens_pos_label.setText(txt)
        except Exception as e:
            self.status_changed.emit(f"Lens read failed: {e}")

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
            self.status_changed.emit("Only a single-frame detection was available (hold still ~2s for averaging).")
        else:
            self.status_changed.emit("No dot detected in both cameras — nothing captured.")
            return

        # 2) rig coordinates: auto from stage or manual fields
        zaber_xyz = None
        stage_um = None
        if self.stage is not None and self.auto_fill_checkbox.isChecked():
            if self.reference_um is None:
                self.status_changed.emit("Set the Reference Pose first (laser on dot, lens at 0).")
                return
            try:
                stage_um = list(self.stage.get_position())
                zaber_xyz = rig_coords_from_stage(stage_um, self.reference_um, self.reference_lens_um)
                self.x_input.setText(f"{zaber_xyz[0]:.4f}")
                self.y_input.setText(f"{zaber_xyz[1]:.4f}")
                self.z_input.setText(f"{zaber_xyz[2]:.4f}")
                self._update_stage_display()
            except Exception as e:
                self.status_changed.emit(f"Stage read failed: {e}")
        if zaber_xyz is None:
            zaber_xyz = self._read_xyz_inputs()
        if zaber_xyz is None:
            self.status_changed.emit("Rig coordinates missing — connect the stage or fill X/Y/Z (mm).")

        # lens sanity: it must not have moved since the reference pose was set
        if self.lens is not None and self.reference_um is not None:
            try:
                lp = float(self.lens.get_position())
                if abs(lp - self.reference_lens_um) > LENS_MOVED_TOL_UM:
                    self.status_changed.emit(
                        f"WARNING: lens moved ({lp:.1f} µm vs {self.reference_lens_um:.1f} µm at reference) "
                        "— z convention broken! Move it back or set a new reference.")
                self._update_lens_display()
            except Exception:
                pass

        new_row = {
            "label": self.label_input.text().strip(),
            "left": left_xyz,
            "zaber": zaber_xyz,
            "stage_um": stage_um,
            "tri_rms": tri_rms,
            "use": True,
            "base": None,
        }

        # 3) motion consistency vs the previous complete capture
        prev = next((r for r in reversed(self.rows) if self._row_valid(r) and r["use"]), None)
        if prev is not None and self._row_valid(new_row):
            R = self.fit_result["T"].R if self.fit_result is not None else None
            msg, ok = motion_check(prev["left"], prev["zaber"],
                                   new_row["left"], new_row["zaber"], R=R)
            self.motion_label.setText(f"Motion check: {msg}")
            self.motion_label.setStyleSheet("color: #007700;" if ok else "color: #cc6600; font-weight: bold;")

        # 4) save images (calibrate_transformation.py-compatible layout)
        if self.save_images_checkbox.isChecked() and self.save_dir is not None \
                and self._last_left is not None and self._last_right is not None:
            try:
                new_row["base"] = self._save_capture_images(self._last_left, self._last_right, zaber_xyz)
            except Exception as e:
                self.status_changed.emit(f"Image save failed: {e}")

        self.rows.append(new_row)
        self._refresh_table()
        self._refit()
        self._autosave_session()
        self.status_changed.emit(f"Captured point #{len(self.rows) - 1:03d}"
                                 + (f" ({new_row['base']})" if new_row["base"] else ""))

    # ---------------- image saving ----------------
    def on_choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Output Folder")
        if not path:
            return
        self.save_dir = Path(path)
        (self.save_dir / "left").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "right").mkdir(exist_ok=True)
        (self.save_dir / "coordinates").mkdir(exist_ok=True)
        self.folder_label.setText(str(self.save_dir))
        self.status_changed.emit(f"Images will be saved to {self.save_dir}")

    def _save_capture_images(self, left, right, zaber_xyz) -> str:
        existing = list((self.save_dir / "left").glob("point_*_left.png"))
        idx = len(existing)
        base = f"point_{idx:04d}"
        cv2.imwrite(str(self.save_dir / "left" / f"{base}_left.png"), left)
        cv2.imwrite(str(self.save_dir / "right" / f"{base}_right.png"), right)
        if zaber_xyz is not None:
            (self.save_dir / "coordinates" / f"{base}.txt").write_text(
                f"{zaber_xyz[0]:.6f} {zaber_xyz[1]:.6f} {zaber_xyz[2]:.6f}\n", encoding="utf-8")
        return base

    # ---------------- transform overlay ----------------
    def on_load_transform(self):
        start = str(ACTIVE_TRANSFORM_PATH) if ACTIVE_TRANSFORM_PATH.exists() else ""
        path, _ = QFileDialog.getOpenFileName(self, "Load Transform JSON", start, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            R = np.asarray(d["R"], float).reshape(3, 3)
            t = np.asarray(d["t"], float).reshape(3)
            self.loaded_T = SE3(R, t)
            self.loaded_T_name = Path(path).stem
            self.transform_src_label.setText(f"loaded: {self.loaded_T_name}")
            self.status_changed.emit(f"Transform '{self.loaded_T_name}' loaded — ZABER overlay active.")
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    # ---------------- 3D view ----------------
    def on_show_3d(self):
        if self.viewer3d is None:
            self.viewer3d = Fit3DDialog(self)
        self._update_3d()
        self.viewer3d.show()
        self.viewer3d.raise_()

    def _update_3d(self):
        if self.viewer3d is None or not self.viewer3d.isVisible():
            if self.viewer3d is None:
                return
        A, B = self._used_AB()
        if self.fit_result is not None and A is not None:
            self.viewer3d.update_data(A, B, self.fit_result["T"], self.fit_result["rms"])
        else:
            self.viewer3d.update_data(A, B, None, None)

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

                self.table.setItem(i, 1, cell(r["label"] or (r.get("base") or "")))
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
            self._update_3d()
            return
        try:
            self.fit_result = fit_pairs(A, B)
        except Exception as e:
            self.fit_label.setText(f"Fit failed: {e}")
            self.hints_label.setText("")
            self._update_3d()
            return

        rms = self.fit_result["rms"]
        scale = self.fit_result["scale"]
        txt = f"Fit: N={len(A)}  RMS={rms:.4f} mm  |  scale check: {scale:.4f}"
        style = "font-weight: bold;"
        if np.isfinite(scale) and abs(scale - 1.0) > SCALE_WARN:
            txt += "  ⚠ scale far from 1 — LEFT (mm) and rig values use different units?"
            style += " color: #cc6600;"
        self.fit_label.setStyleSheet(style)
        self.fit_label.setText(txt)

        hints = geometry_hints(B)
        self.hints_label.setText("Hints: " + " | ".join(hints))
        self._update_3d()

    # ---------------- session persistence ----------------
    def _autosave_session(self):
        try:
            payload = {
                "rows": self.rows,
                "reference_um": self.reference_um,
                "reference_lens_um": self.reference_lens_um,
                "save_dir": str(self.save_dir) if self.save_dir else None,
                "saved_at": datetime.datetime.now().isoformat(),
            }
            with open(SESSION_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
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
            self.reference_um = d.get("reference_um")
            self.reference_lens_um = float(d.get("reference_lens_um", 0.0))
            if self.reference_um is not None:
                r = self.reference_um
                self.ref_label.setText(f"Reference: ({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}) µm | "
                                       f"lens {self.reference_lens_um:.1f} µm")
            sd = d.get("save_dir")
            if sd:
                self.save_dir = Path(sd)
                self.folder_label.setText(sd)
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
            "description": "Transformation from left camera frame to zaber rig coordinate system",
            "R": T.R.tolist(),
            "t": T.t.tolist(),
            "source": {
                "method": "umeyama_fit",
                "num_points": int(len(A)),
                "rms_error": self.fit_result["rms"],
                "scale_check": self.fit_result["scale"],
            },
            "convention": {
                "frame": "moving rig frame, axes parallel to stage axes",
                "xy_zero": "laser beam axis",
                "z_zero": "laser focus with eye lens at 0 um (dot z at reference = lens reading in mm)",
                "rig_coords": "x,y = -(stage - reference); z = lens_ref_mm - delta_stage_z  [mm]",
                "reference_pose_um": self.reference_um,
                "reference_lens_um": self.reference_lens_um,
            },
            "data": {
                "all_pairs": [{"left": r["left"], "input": r["zaber"], "use": r["use"],
                               "label": r["label"], "tri_rms": r["tri_rms"],
                               "stage_um": r.get("stage_um"), "base": r.get("base")} for r in self.rows],
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
               f"N={obj['source']['num_points']}, RMS={obj['source']['rms_error']:.4f} mm\n"
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
            for dev in (self.stage, self.lens):
                if dev is not None:
                    try:
                        dev.close()
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
    w.resize(1220, 1050)
    w.show()
    sys.exit(app.exec_())
