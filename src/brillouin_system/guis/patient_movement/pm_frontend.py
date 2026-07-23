"""
Patient-movement GUI frontend.

A standalone tool (separate from the main Brillouin GUI) for tracking how the
patient's cornea moves axially:

  - two Allied Vision cameras via the SAME EyeTrackerController / worker
    process as the main GUI (pupil3D estimates included)
  - reflection plane finding with the SAME finder + config dialog
  - cornea sweep tracking (CorneaTracker): find the plane, then sweep
    +- amplitude around it; each up/down crossing pair gives a latency-bias-
    free surface position, shown live on a strip chart
  - laser XY calibration (same workflow as the main GUI's
    "Cal. Laser Offset" button)

Run via guis/patient_movement/main.py.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
    QWidget,
)

from brillouin_system.eye_tracker.calibrate_camera_laser_position.calib_rig_laser_position import (
    LaserOffset, load_laser_coord_system_from_toml)
from brillouin_system.eye_tracker.eye_position.coordinates import RigCoord
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config_gui import EyeTrackerConfigDialog
from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults, get_eye_tracker_results
from brillouin_system.guis.human_interface.eye_tracker_controller import EyeTrackerController
from brillouin_system.guis.patient_movement.pm_backend import PmBackend
from brillouin_system.logging_utils.logging_setup import get_logger, logging_fmt_gui
from brillouin_system.logging_utils.qt_log_bridge import QtLogBridge
from brillouin_system.logging_utils.qt_log_handler import QtTextEditHandler
from brillouin_system.patient_movement_analysis.tracking_config.tracking_config import (
    save_tracking_config, tracking_config)
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult
from brillouin_system.scan_managers.scanning_config.scanning_config_gui import AxialScanningConfigDialog

log = get_logger(__name__)


class _Bridge(QtCore.QObject):
    """Signals emitted from worker threads back into the GUI thread."""
    reflection_done = pyqtSignal(object)          # ReflectionResult
    laser_calib_done = pyqtSignal(object)         # LaserOffset
    laser_calib_failed = pyqtSignal(str)
    tracking_stopped = pyqtSignal(object, object)  # points, estimates
    op_failed = pyqtSignal(str)
    positions_changed = pyqtSignal()               # zaber positions need re-reading


class PmFrontend(QWidget):

    # eye tracker control (cross-thread via queued signals)
    set_et_config = pyqtSignal(object)
    request_et_start = pyqtSignal()
    request_et_stop = pyqtSignal()
    request_eye_shutdown = pyqtSignal()

    def __init__(self, use_backend_dummy: bool = True, use_eye_tracker_dummy: bool = True):
        super().__init__()
        self.setWindowTitle("Patient Movement Tracker")

        # --- logging into the GUI (same pattern as the main GUI) ---
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(120)
        self.log_bridge = QtLogBridge()
        self.log_bridge.message.connect(self.log_view.append)
        self.qt_handler = QtTextEditHandler(self.log_bridge)
        self.qt_handler.setFormatter(logging_fmt_gui)
        lg = get_logger()
        if self.qt_handler not in lg.handlers:
            lg.addHandler(self.qt_handler)
        lg.setLevel(logging.INFO)

        # --- backend (NI + Zabers) ---
        self.backend = PmBackend(use_dummy=use_backend_dummy)

        # --- state ---
        self._reflection_result: ReflectionResult | None = None
        self._latest_pupil3d = None
        self._latest_pupil_t = 0.0
        self._latest_et_results: EyeTrackerResults | None = None
        self._move_lock = threading.Lock()   # serializes manual zaber moves
        self._zaber_lens_um: float | None = None
        try:
            self._laser_offset: LaserOffset = load_laser_coord_system_from_toml()
        except Exception:
            self._laser_offset = LaserOffset(0.0, 0.0, 0.0)
            log.warning("[Frontend] No laser offset.toml found — laser map uses (0,0,0). "
                        "Run the laser XY calibration.")
        self.laser_focus_position: RigCoord | None = None
        self._point_q: queue.Queue = queue.Queue()
        self._est_q: queue.Queue = queue.Queue()
        self._track_t0: float | None = None
        self._plot_up_t: deque = deque(maxlen=5000)
        self._plot_up_z: deque = deque(maxlen=5000)
        self._plot_down_t: deque = deque(maxlen=5000)
        self._plot_down_z: deque = deque(maxlen=5000)
        self._plot_est_t: deque = deque(maxlen=5000)
        self._plot_est_z: deque = deque(maxlen=5000)
        self._est_times: deque = deque(maxlen=20)   # for rate display
        self._n_passes = 0
        self._n_misses = 0
        self._pupil_track: list[dict] = []
        self._last_points: list = []
        self._last_estimates: list = []

        self.bridge = _Bridge()
        self.bridge.reflection_done.connect(self._on_reflection_done)
        self.bridge.laser_calib_done.connect(self._on_laser_calib_done)
        self.bridge.laser_calib_failed.connect(self._on_laser_calib_failed)
        self.bridge.tracking_stopped.connect(self._on_tracking_stopped)
        self.bridge.op_failed.connect(self._on_op_failed)
        self.bridge.positions_changed.connect(self._refresh_positions)

        self._init_ui()
        self._refresh_positions()

        # --- eye tracker (same controller/worker as the main GUI) ---
        self.eye_thread = QThread(self)
        self.eye_ctrl = EyeTrackerController(use_dummy=use_eye_tracker_dummy)
        self.eye_ctrl.moveToThread(self.eye_thread)
        self.eye_thread.started.connect(self.eye_ctrl.start)
        self.eye_ctrl.frames_ready.connect(self._on_eye_frames_ready)
        self.set_et_config.connect(self.eye_ctrl.send_config)
        self.request_et_start.connect(self.eye_ctrl.start)
        self.request_et_stop.connect(self.eye_ctrl.stop)
        self.request_eye_shutdown.connect(self.eye_ctrl.shutdown)
        self.eye_thread.start()

        # --- GUI update timer (drains tracker queues) ---
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.timeout.connect(self._drain_track_queues)
        self._update_timer.start(50)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _init_ui(self):
        outer = QHBoxLayout()
        self.setLayout(outer)

        # LEFT: controls
        left = QVBoxLayout()
        left.addWidget(self._group_reflection())
        left.addWidget(self._group_tracking())
        left.addWidget(self._group_laser_calibration())
        left.addWidget(self._group_manual_zabers())
        left.addWidget(self._group_log())
        left.addStretch()
        outer.addLayout(left, 0)

        # RIGHT: cameras + strip chart
        right = QVBoxLayout()
        right.addWidget(self._group_eye_tracking(), 2)
        right.addWidget(self._group_strip_chart(), 1)
        outer.addLayout(right, 1)

    def _group_reflection(self) -> QGroupBox:
        g = QGroupBox("Reflection Plane")
        v = QVBoxLayout()

        row = QHBoxLayout()
        self.btn_find = QPushButton("Find Reflection Plane")
        self.btn_find.clicked.connect(self._on_find_reflection_clicked)
        self.btn_scan_settings = QPushButton("Settings…")
        self.btn_scan_settings.clicked.connect(self._open_axial_settings_dialog)
        row.addWidget(self.btn_find)
        row.addWidget(self.btn_scan_settings)
        v.addLayout(row)

        self.lbl_reflection = QLabel("Plane: not found yet")
        v.addWidget(self.lbl_reflection)
        g.setLayout(v)
        return g

    def _group_tracking(self) -> QGroupBox:
        g = QGroupBox("Cornea Tracking")
        v = QVBoxLayout()
        tc = tracking_config.get()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Sweep ± [µm]"))
        self.spin_amplitude = QDoubleSpinBox()
        self.spin_amplitude.setRange(5.0, 2000.0)
        self.spin_amplitude.setValue(tc.sweep_amplitude_um)
        self.spin_amplitude.setDecimals(0)
        row1.addWidget(self.spin_amplitude)
        row1.addWidget(QLabel("Speed [µm/s]"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(100.0, 10000.0)
        self.spin_speed.setValue(tc.sweep_speed_um_s)
        self.spin_speed.setDecimals(0)
        row1.addWidget(self.spin_speed)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        self.chk_recenter = QCheckBox("Re-center window on surface")
        self.chk_recenter.setChecked(tc.recenter)
        row2.addWidget(self.chk_recenter)
        self.btn_save_track_cfg = QPushButton("Save as default")
        self.btn_save_track_cfg.clicked.connect(self._save_tracking_defaults)
        row2.addWidget(self.btn_save_track_cfg)
        v.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_track_start = QPushButton("Start Tracking")
        self.btn_track_start.setEnabled(False)
        self.btn_track_start.clicked.connect(self._on_start_tracking_clicked)
        self.btn_track_stop = QPushButton("Stop")
        self.btn_track_stop.setEnabled(False)
        self.btn_track_stop.clicked.connect(self._on_stop_tracking_clicked)
        self.btn_track_save = QPushButton("Save Track…")
        self.btn_track_save.setEnabled(False)
        self.btn_track_save.clicked.connect(self._on_save_track_clicked)
        row3.addWidget(self.btn_track_start)
        row3.addWidget(self.btn_track_stop)
        row3.addWidget(self.btn_track_save)
        v.addLayout(row3)

        self.lbl_track_status = QLabel("Tracking: idle")
        v.addWidget(self.lbl_track_status)
        g.setLayout(v)
        return g

    def _group_laser_calibration(self) -> QGroupBox:
        g = QGroupBox("Laser XY Calibration")
        v = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_laser_calib = QPushButton("Cal. Laser Offset")
        self.btn_laser_calib.clicked.connect(self._on_laser_calib_clicked)
        self.btn_laser_cancel = QPushButton("Cancel")
        self.btn_laser_cancel.setEnabled(False)
        self.btn_laser_cancel.clicked.connect(self.backend.request_cancel)
        row.addWidget(self.btn_laser_calib)
        row.addWidget(self.btn_laser_cancel)
        v.addLayout(row)
        self.lbl_laser_offset = QLabel("Offset: —")
        v.addWidget(self.lbl_laser_offset)
        g.setLayout(v)
        return g

    def _group_manual_zabers(self) -> QGroupBox:
        """All four axes (eye lens + rig stage x/y/z), same directions and
        sign conventions as the main GUI's 'Manually Move Zabers' group."""
        g = QGroupBox("Manually Move Zabers")
        layout = QFormLayout()

        def make_row(label: QLabel, step_input: QLineEdit, *buttons):
            row = QHBoxLayout()
            row.addWidget(step_input)
            for btn in buttons:
                row.addWidget(btn)
            row.addStretch()
            return label, row

        # LENS AXIS
        self.lens_step_input = QLineEdit("100")
        self.lens_step_input.setFixedWidth(60)
        lens_back = QPushButton("← Back")
        lens_fwd = QPushButton("→ Forward")
        self.lbl_lens_pos = QLabel("Lens ? µm")
        layout.addRow(*make_row(self.lbl_lens_pos, self.lens_step_input, lens_back, lens_fwd))
        lens_back.clicked.connect(lambda: self._move_lens_rel(-float(self.lens_step_input.text())))
        lens_fwd.clicked.connect(lambda: self._move_lens_rel(+float(self.lens_step_input.text())))

        # STAGE Z
        self.z_step_input = QLineEdit("100")
        self.z_step_input.setFixedWidth(60)
        z_back = QPushButton("← Back")
        z_fwd = QPushButton("→ Forward")
        self.lbl_stage_z = QLabel("Stage Z ? µm")
        layout.addRow(*make_row(self.lbl_stage_z, self.z_step_input, z_back, z_fwd))
        z_back.clicked.connect(lambda: self._move_stage_rel(dz=-float(self.z_step_input.text())))
        z_fwd.clicked.connect(lambda: self._move_stage_rel(dz=+float(self.z_step_input.text())))

        # STAGE X
        self.x_step_input = QLineEdit("100")
        self.x_step_input.setFixedWidth(60)
        x_left = QPushButton("← Left")
        x_right = QPushButton("→ Right")
        self.lbl_stage_x = QLabel("Stage X ? µm")
        layout.addRow(*make_row(self.lbl_stage_x, self.x_step_input, x_left, x_right))
        x_left.clicked.connect(lambda: self._move_stage_rel(dx=+float(self.x_step_input.text())))
        x_right.clicked.connect(lambda: self._move_stage_rel(dx=-float(self.x_step_input.text())))

        # STAGE Y
        self.y_step_input = QLineEdit("100")
        self.y_step_input.setFixedWidth(60)
        y_up = QPushButton("↑ Up")
        y_down = QPushButton("↓ Down")
        self.lbl_stage_y = QLabel("Stage Y ? µm")
        layout.addRow(*make_row(self.lbl_stage_y, self.y_step_input, y_up, y_down))
        y_up.clicked.connect(lambda: self._move_stage_rel(dy=+float(self.y_step_input.text())))
        y_down.clicked.connect(lambda: self._move_stage_rel(dy=-float(self.y_step_input.text())))

        # Move laser to (R, phi) using eye tracking (ABSOLUTE target)
        self.xy_r_input = QLineEdit("0.0")
        self.xy_r_input.setFixedWidth(60)
        self.xy_r_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 4))   # mm
        self.xy_phi_input = QLineEdit("0.0")
        self.xy_phi_input.setFixedWidth(60)
        self.xy_phi_input.setValidator(QDoubleValidator(-3600.0, 3600.0, 3))  # deg
        xy_move = QPushButton("Move XY")
        xy_move.setFixedWidth(70)
        xy_move.clicked.connect(self._on_move_xy_polar_clicked)
        xy_row = QHBoxLayout()
        xy_row.addWidget(QLabel("R [mm]:"))
        xy_row.addWidget(self.xy_r_input)
        xy_row.addSpacing(6)
        xy_row.addWidget(QLabel("phi [deg]:"))
        xy_row.addWidget(self.xy_phi_input)
        xy_row.addSpacing(6)
        xy_row.addWidget(xy_move)
        xy_row.addStretch()
        layout.addRow(xy_row)

        # Move lens Z to reach a target Δc (uses current Δc from eye tracking)
        self.dc_input = QLineEdit("2.0")
        self.dc_input.setFixedWidth(60)
        self.dc_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 4))     # mm
        dc_move = QPushButton("Move Z")
        dc_move.setFixedWidth(70)
        dc_move.clicked.connect(self._on_move_z_by_dc_clicked)
        dc_row = QHBoxLayout()
        dc_row.addWidget(QLabel("Δc [mm]:"))
        dc_row.addWidget(self.dc_input)
        dc_row.addSpacing(6)
        dc_row.addWidget(dc_move)
        dc_row.addStretch()
        layout.addRow(dc_row)

        btn_read = QPushButton("Read Positions")
        btn_read.clicked.connect(self._refresh_positions)
        layout.addRow(btn_read)

        g.setLayout(layout)
        return g

    def _group_log(self) -> QGroupBox:
        g = QGroupBox("Log Output")
        v = QVBoxLayout()
        v.addWidget(self.log_view)
        g.setLayout(v)
        return g

    def _group_eye_tracking(self) -> QGroupBox:
        g = QGroupBox("Eye Tracking (Allied Vision)")
        v = QVBoxLayout()

        self.eye_glw = pg.GraphicsLayoutWidget()
        self.eye_glw.ci.setContentsMargins(4, 4, 4, 4)
        self.eye_glw.ci.setSpacing(4)
        self.eye_img = []
        for col in range(2):
            vb = pg.ViewBox(lockAspect=True, enableMenu=False)
            vb.invertY(True)
            img = pg.ImageItem(autoDownsample=True)
            vb.addItem(img)
            vb.setBorder((80, 80, 80))
            self.eye_glw.ci.addItem(vb, row=0, col=col)
            self.eye_img.append(img)

        # Laser position map (same as the main GUI): polar grid in pupil
        # coordinates [mm], moving laser dot, x/y/z/Δc text readout.
        map_vb = pg.ViewBox(lockAspect=True, enableMenu=False)
        map_vb.setRange(xRange=(-4, 4), yRange=(-4, 4))
        map_vb.setBorder((80, 80, 80))
        angles = np.linspace(0, 2 * np.pi, 720)
        for r in range(1, 5):
            map_vb.addItem(pg.PlotDataItem(r * np.cos(angles), r * np.sin(angles),
                                           pen=pg.mkPen(150, 150, 150)))
        for angle_deg in (0, 45, 90, 135, 180, 225, 270, 315):
            theta = np.deg2rad(angle_deg)
            map_vb.addItem(pg.PlotDataItem(
                [np.cos(theta), 4 * np.cos(theta)], [np.sin(theta), 4 * np.sin(theta)],
                pen=pg.mkPen(150, 150, 150, style=QtCore.Qt.DashLine)))
        self.laser_point = pg.ScatterPlotItem([0.0], [0.0], size=12,
                                              brush=pg.mkBrush("r"), pen=pg.mkPen("r"))
        map_vb.addItem(self.laser_point)
        self.laser_pos_text = pg.TextItem(text="x=\ny=\nz=\nΔc=", anchor=(1, 0))
        map_vb.addItem(self.laser_pos_text)
        self.laser_pos_text.setPos(4, 4)
        self.eye_glw.ci.addItem(map_vb, row=1, col=0)
        grid = self.eye_glw.ci.layout
        grid.setRowStretchFactor(0, 2)
        grid.setRowStretchFactor(1, 1)

        v.addWidget(self.eye_glw)

        row = QHBoxLayout()
        self.lbl_pupil = QLabel("Pupil: —")
        row.addWidget(self.lbl_pupil)
        row.addStretch()
        self.btn_et_config = QPushButton("ET Config…")
        self.btn_et_config.clicked.connect(self._open_et_config_dialog)
        self.btn_et_restart = QPushButton("ReStart Cameras")
        self.btn_et_restart.clicked.connect(self._restart_eye_tracker)
        row.addWidget(self.btn_et_config)
        row.addWidget(self.btn_et_restart)
        v.addLayout(row)

        g.setLayout(v)
        g.setMinimumWidth(700)
        g.setMinimumHeight(420)
        return g

    def _group_strip_chart(self) -> QGroupBox:
        g = QGroupBox("Cornea Position (live)")
        v = QVBoxLayout()
        self.track_plot = pg.PlotWidget()
        self.track_plot.setLabel("bottom", "time (s)")
        self.track_plot.setLabel("left", "surface z (µm)")
        self.track_plot.showGrid(x=True, y=True, alpha=0.25)
        self.track_plot.addLegend(offset=(10, 10))
        self.scatter_up = self.track_plot.plot(
            [], [], pen=None, symbol="t1", symbolSize=6,
            symbolBrush=(80, 160, 255, 150), name="up crossings")
        self.scatter_down = self.track_plot.plot(
            [], [], pen=None, symbol="t", symbolSize=6,
            symbolBrush=(255, 170, 60, 150), name="down crossings")
        self.curve_est = self.track_plot.plot(
            [], [], pen=pg.mkPen((120, 255, 120), width=2),
            symbol="o", symbolSize=5, symbolBrush=(120, 255, 120),
            name="pair estimate (bias-free)")
        v.addWidget(self.track_plot)
        g.setLayout(v)
        return g

    # ------------------------------------------------------------------ #
    # reflection plane
    # ------------------------------------------------------------------ #

    def _on_find_reflection_clicked(self):
        self.btn_find.setEnabled(False)
        self.btn_track_start.setEnabled(False)
        self.lbl_reflection.setText("Plane: searching…")

        def _work():
            try:
                res = self.backend.find_reflection_plane(is_go_forwards=True)
                self.bridge.reflection_done.emit(res)
            except Exception as e:
                self.bridge.op_failed.emit(f"Reflection find failed: {e}")

        threading.Thread(target=_work, daemon=True).start()

    def _on_reflection_done(self, res: ReflectionResult):
        self.btn_find.setEnabled(True)
        self._reflection_result = res
        if res.found and res.event_z_um is not None and np.isfinite(res.event_z_um):
            self.lbl_reflection.setText(
                f"Plane: z = {res.event_z_um:.1f} µm   "
                f"(peak {res.peak_value:.2f} V, {res.n_samples_above} samples)")
            self.btn_track_start.setEnabled(True)
            self._refresh_positions()
        else:
            self.lbl_reflection.setText(
                f"Plane: NOT FOUND ({res.n_rejected_intervals} noise intervals rejected)")

    def _open_axial_settings_dialog(self):
        dlg = AxialScanningConfigDialog(
            on_apply=lambda cfg: self.backend.update_axial_config(cfg), parent=self)
        dlg.exec_()

    # ------------------------------------------------------------------ #
    # tracking
    # ------------------------------------------------------------------ #

    def _apply_tracking_settings(self):
        tracking_config.update(
            sweep_amplitude_um=float(self.spin_amplitude.value()),
            sweep_speed_um_s=float(self.spin_speed.value()),
            recenter=bool(self.chk_recenter.isChecked()),
        )
        self.backend.update_tracking_config(tracking_config.get())

    def _save_tracking_defaults(self):
        self._apply_tracking_settings()
        save_tracking_config(tracking_config)
        log.info("[Frontend] Tracking settings saved as default.")

    def _on_start_tracking_clicked(self):
        if self._reflection_result is None or not self._reflection_result.found:
            QMessageBox.warning(self, "Tracking", "Find the reflection plane first.")
            return
        self._apply_tracking_settings()

        # reset live plot + session buffers
        for d in (self._plot_up_t, self._plot_up_z, self._plot_down_t,
                  self._plot_down_z, self._plot_est_t, self._plot_est_z,
                  self._est_times):
            d.clear()
        self._n_passes = 0
        self._n_misses = 0
        self._pupil_track = []
        self._track_t0 = time.perf_counter()

        try:
            self.backend.start_tracking(
                float(self._reflection_result.event_z_um),
                on_point=self._point_q.put,
                on_estimate=self._est_q.put,
            )
        except Exception as e:
            QMessageBox.critical(self, "Tracking", str(e))
            return

        self.btn_track_start.setEnabled(False)
        self.btn_track_stop.setEnabled(True)
        self.btn_track_save.setEnabled(False)
        self.btn_find.setEnabled(False)
        self.btn_laser_calib.setEnabled(False)
        self.lbl_track_status.setText("Tracking: running…")

    def _on_stop_tracking_clicked(self):
        self.btn_track_stop.setEnabled(False)

        def _work():
            try:
                points, estimates = self.backend.stop_tracking()
                self.bridge.tracking_stopped.emit(points, estimates)
            except Exception as e:
                self.bridge.op_failed.emit(f"Stop tracking failed: {e}")

        threading.Thread(target=_work, daemon=True).start()

    def _on_tracking_stopped(self, points, estimates):
        self._last_points = points
        self._last_estimates = estimates
        err = self.backend.get_tracking_error()
        n_found = sum(1 for p in points if p.found)
        status = (f"Tracking: stopped — {len(points)} passes, {n_found} found, "
                  f"{len(estimates)} estimates")
        if err:
            status += f"  (ERROR: {err})"
        self.lbl_track_status.setText(status)
        self.btn_track_start.setEnabled(True)
        self.btn_track_stop.setEnabled(False)
        self.btn_track_save.setEnabled(len(points) > 0)
        self.btn_find.setEnabled(True)
        self.btn_laser_calib.setEnabled(True)

    def _on_save_track_clicked(self):
        default = str(Path.home() / f"cornea_track_{time.strftime('%Y%m%d_%H%M%S')}.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save track", default, "JSON files (*.json)")
        if not path:
            return
        try:
            self.backend.save_track(
                path, self._last_points, self._last_estimates,
                pupil_track=self._pupil_track,
                reflection_result=self._reflection_result,
            )
        except Exception as e:
            QMessageBox.critical(self, "Save", str(e))

    def _drain_track_queues(self):
        if self._track_t0 is None:
            return
        updated = False
        while True:
            try:
                p = self._point_q.get_nowait()
            except queue.Empty:
                break
            self._n_passes += 1
            if p.found:
                t = p.t_perf - self._track_t0
                if p.direction == "up":
                    self._plot_up_t.append(t)
                    self._plot_up_z.append(p.z_um)
                else:
                    self._plot_down_t.append(t)
                    self._plot_down_z.append(p.z_um)
            else:
                self._n_misses += 1
            updated = True

        last_est = None
        while True:
            try:
                e = self._est_q.get_nowait()
            except queue.Empty:
                break
            self._plot_est_t.append(e.t_perf - self._track_t0)
            self._plot_est_z.append(e.z_um)
            self._est_times.append(e.t_perf)
            last_est = e
            updated = True

        if not updated:
            return

        self.scatter_up.setData(list(self._plot_up_t), list(self._plot_up_z))
        self.scatter_down.setData(list(self._plot_down_t), list(self._plot_down_z))
        self.curve_est.setData(list(self._plot_est_t), list(self._plot_est_z))

        if self.backend.is_tracking():
            rate = 0.0
            if len(self._est_times) >= 2:
                span = self._est_times[-1] - self._est_times[0]
                if span > 0:
                    rate = (len(self._est_times) - 1) / span
            z_txt = f"{last_est.z_um:.1f} µm" if last_est is not None else "—"
            self.lbl_track_status.setText(
                f"Tracking: running — z = {z_txt}   {rate:.1f} est/s   "
                f"{self._n_passes} passes, {self._n_misses} misses")

    # ------------------------------------------------------------------ #
    # laser calibration (same two-stage workflow as the main GUI)
    # ------------------------------------------------------------------ #

    def _on_laser_calib_clicked(self):
        pupil = self._latest_pupil3d
        age = time.monotonic() - self._latest_pupil_t
        if pupil is None or pupil.center_ref is None or age > 1.0:
            QMessageBox.warning(
                self, "Laser Calibration",
                "No recent pupil estimate from the eye tracker — "
                "start the cameras and make sure the pupil is detected.")
            return

        # Stage 1: recenter the rig on the pupil (pupil center is in mm).
        x_um = float(pupil.center_ref[0]) * 1000.0
        y_um = float(pupil.center_ref[1]) * 1000.0
        z_um = float(pupil.center_ref[2]) * 1000.0
        try:
            lens_um = float(self.backend.zaber_eye_lens.get_position())
        except Exception as e:
            QMessageBox.critical(self, "Laser Calibration", f"Lens position read failed: {e}")
            return

        self.btn_laser_calib.setEnabled(False)
        self.btn_laser_cancel.setEnabled(True)
        self.btn_find.setEnabled(False)
        self.btn_track_start.setEnabled(False)
        self.lbl_laser_offset.setText("Offset: calibrating…")
        log.info(f"[Laser Calib] Recentering rig on pupil "
                 f"(dx={x_um:.0f}, dy={y_um:.0f}, dz={z_um - lens_um:.0f} µm), then calibrating.")

        def _work():
            try:
                self.backend.zaber_hi.move_rel(dx=x_um, dy=y_um, dz=None)
                self.backend.zaber_hi.move_rel(dx=None, dy=None, dz=z_um - lens_um)
                time.sleep(2.0)
                offset = self.backend.run_laser_xy_calibration()
                self.bridge.laser_calib_done.emit(offset)
            except Exception as e:
                self.bridge.laser_calib_failed.emit(str(e))

        threading.Thread(target=_work, daemon=True).start()

    def _on_laser_calib_done(self, offset):
        self.lbl_laser_offset.setText(
            f"Offset: dx={offset.dx:.3f}, dy={offset.dy:.3f}, dz={offset.dz:.3f}")
        # use the fresh offset for the laser map immediately
        self._laser_offset = offset
        self._refresh_positions()
        self._laser_calib_buttons_reset()

    def _on_laser_calib_failed(self, msg: str):
        self.lbl_laser_offset.setText(f"Offset: FAILED — {msg}")
        self._laser_calib_buttons_reset()

    def _laser_calib_buttons_reset(self):
        self.btn_laser_calib.setEnabled(True)
        self.btn_laser_cancel.setEnabled(False)
        self.btn_find.setEnabled(True)
        self.btn_track_start.setEnabled(
            self._reflection_result is not None and self._reflection_result.found)

    # ------------------------------------------------------------------ #
    # eye tracker
    # ------------------------------------------------------------------ #

    @QtCore.pyqtSlot(object, object, dict)
    def _on_eye_frames_ready(self, left, right, meta):
        self.eye_img[0].setImage(left, autoLevels=True)
        self.eye_img[1].setImage(right, autoLevels=True)

        # Full eye-tracker result incl. laser position in pupil coordinates
        # (needs laser_focus_position from the lens z + calibrated offset).
        if self.laser_focus_position is not None:
            self._latest_et_results = get_eye_tracker_results(
                left=left, right=right, meta=meta,
                laser_focus_position=self.laser_focus_position)
        else:
            self._latest_et_results = None

        pupil3d = meta.get("pupil3D")
        if pupil3d is not None and pupil3d.center_ref is not None:
            self._latest_pupil3d = pupil3d
            self._latest_pupil_t = time.monotonic()
            c = pupil3d.center_ref
            self.lbl_pupil.setText(
                f"Pupil (rig): x={c[0]:.2f}  y={c[1]:.2f}  z={c[2]:.2f} mm")
            if self.backend.is_tracking():
                self._pupil_track.append({
                    "t_perf": time.perf_counter(),
                    "center_ref_mm": [float(v) for v in c],
                    "radius_mm": float(pupil3d.radius) if pupil3d.radius is not None else None,
                })
        else:
            self.lbl_pupil.setText("Pupil: —")

        # Laser map dot + text
        res = self._latest_et_results
        if res is not None and res.laser_position is not None:
            x, y, z = res.laser_position
            if np.isfinite(x) and np.isfinite(y):
                self.laser_point.setData([float(x)], [float(y)])
                self.laser_point.setVisible(True)
            dc = res.delta_laser_corner

            def fmt(v):
                return f"{v:6.3f}" if v is not None else " " * 6

            self.laser_pos_text.setText(f"x={fmt(x)}\ny={fmt(y)}\nz={fmt(z)}\nΔc={fmt(dc)}")
        else:
            self.laser_point.setVisible(False)
            self.laser_pos_text.setText("x=\ny=\nz=\nΔc=")

    def _recent_et_result(self, max_age_s: float = 0.5) -> EyeTrackerResults | None:
        """Latest eye-tracker result with a valid laser position, if fresh."""
        res = self._latest_et_results
        if res is None or res.laser_position is None:
            return None
        if (time.monotonic() - self._latest_pupil_t) > max_age_s:
            return None
        return res

    def _open_et_config_dialog(self):
        def _on_apply(cfg: EyeTrackerConfig):
            try:
                self.set_et_config.emit(cfg)
                log.info("[Frontend] Sent new eye tracker config.")
            except Exception as e:
                log.error(f"[Frontend] ET config failed: {e}")

        dlg = EyeTrackerConfigDialog(on_apply=_on_apply, parent=self)
        dlg.exec_()

    def _restart_eye_tracker(self):
        self.request_et_stop.emit()
        self.request_et_start.emit()

    # ------------------------------------------------------------------ #
    # manual zabers + laser positioning
    # ------------------------------------------------------------------ #

    def _move_lens_rel(self, delta_um: float):
        if self.backend.is_tracking():
            QMessageBox.warning(self, "Lens", "Stop tracking before moving the lens manually.")
            return

        def _work():
            with self._move_lock:
                try:
                    self.backend.zaber_eye_lens.move_rel(float(delta_um))
                except Exception as e:
                    self.bridge.op_failed.emit(f"Lens move failed: {e}")
            self.bridge.positions_changed.emit()

        threading.Thread(target=_work, daemon=True).start()

    def _move_stage_rel(self, dx: float | None = None, dy: float | None = None,
                        dz: float | None = None):
        def _work():
            with self._move_lock:
                try:
                    self.backend.zaber_hi.move_rel(dx=dx, dy=dy, dz=dz)
                except Exception as e:
                    self.bridge.op_failed.emit(f"Stage move failed: {e}")
            self.bridge.positions_changed.emit()

        threading.Thread(target=_work, daemon=True).start()

    def _on_move_xy_polar_clicked(self):
        """Move the rig stage so the laser lands at the ABSOLUTE (R, phi)
        position in laser/pupil coordinates. Same math and sign convention as
        the main GUI (stage +X moves the laser -X)."""
        res = self._recent_et_result()
        if res is None:
            QMessageBox.warning(self, "Move XY",
                                "No recent eye-tracker result with a laser position — "
                                "check cameras / pupil detection / laser offset.")
            return
        try:
            x_laser = float(res.laser_position[0])
            y_laser = float(res.laser_position[1])
            r_mm = float(self.xy_r_input.text())
            phi = np.deg2rad(float(self.xy_phi_input.text()))

            dx_um = (r_mm * np.cos(phi) - x_laser) * 1000.0
            dy_um = (r_mm * np.sin(phi) - y_laser) * 1000.0

            max_move_um = 3500.0
            move_mag_um = float(np.hypot(dx_um, dy_um))
            if move_mag_um > max_move_um:
                scale = max_move_um / move_mag_um
                dx_um *= scale
                dy_um *= scale
                log.warning(f"[Move XY] Requested {move_mag_um / 1000:.3f} mm exceeds "
                            f"3.5 mm limit — clamped (scale {scale:.3f}).")

            log.info(f"[Move XY] laser ({x_laser:.3f}, {y_laser:.3f}) mm -> "
                     f"R={r_mm:.3f} mm phi={np.rad2deg(phi):.1f} deg "
                     f"(stage dx={-dx_um:.0f}, dy={dy_um:.0f} µm)")
            self._move_stage_rel(dx=-dx_um, dy=dy_um)
        except Exception as e:
            log.exception(f"[Move XY] Failed: {e}")

    def _on_move_z_by_dc_clicked(self):
        """Move the eye lens so Δc (laser-to-cornea distance) reaches the
        target value. Same convention as the main GUI (lens moves -dz)."""
        if self.backend.is_tracking():
            QMessageBox.warning(self, "Move Z", "Stop tracking before moving the lens.")
            return
        res = self._recent_et_result()
        if res is None or res.delta_laser_corner is None:
            QMessageBox.warning(self, "Move Z",
                                "No recent eye-tracker result with Δc — "
                                "check cameras / pupil detection.")
            return
        try:
            dc_current = float(res.delta_laser_corner)
            dc_target = float(self.dc_input.text())
            dz_um = float(np.clip((dc_target - dc_current) * 1000.0, -2000.0, 2000.0))
            log.info(f"[Move Z Δc] current={dc_current:.3f} target={dc_target:.3f} mm "
                     f"-> lens {-dz_um:.0f} µm")
            self._move_lens_rel(-dz_um)
        except Exception as e:
            log.exception(f"[Move Z Δc] Failed: {e}")

    def _refresh_positions(self):
        """Read all four zaber positions, update labels and the laser focus
        position (rig coords) used by the eye-tracker laser display."""
        try:
            pos = float(self.backend.zaber_eye_lens.get_position())
            self._zaber_lens_um = pos
            self.lbl_lens_pos.setText(f"Lens {pos:.2f} µm")
            self.laser_focus_position = RigCoord(
                x=self._laser_offset.dx / 1000.0,
                y=self._laser_offset.dy / 1000.0,
                z=pos / 1000.0 + self._laser_offset.dz / 1000.0,
            )
        except Exception:
            self.lbl_lens_pos.setText("Lens ? µm")
        try:
            x, y, z = self.backend.zaber_hi.get_position()
            self.lbl_stage_x.setText(f"Stage X {x:.2f} µm")
            self.lbl_stage_y.setText(f"Stage Y {y:.2f} µm")
            self.lbl_stage_z.setText(f"Stage Z {z:.2f} µm")
        except Exception:
            self.lbl_stage_x.setText("Stage X ? µm")
            self.lbl_stage_y.setText("Stage Y ? µm")
            self.lbl_stage_z.setText("Stage Z ? µm")

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #

    def _on_op_failed(self, msg: str):
        log.error(f"[Frontend] {msg}")
        self.btn_find.setEnabled(True)
        if not self.backend.is_tracking():
            self.btn_track_stop.setEnabled(False)
            self.btn_track_start.setEnabled(
                self._reflection_result is not None and self._reflection_result.found)

    def closeEvent(self, event):
        try:
            if self.backend.is_tracking():
                self.backend.stop_tracking()
        except Exception:
            pass
        try:
            self.request_eye_shutdown.emit()
            self.eye_thread.quit()
            self.eye_thread.wait(3000)
        except Exception:
            pass
        # Last resort: the worker process is non-daemon and would keep the
        # interpreter alive if the shutdown handshake did not complete.
        try:
            proc = self.eye_ctrl.proxy.proc
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.backend.close()
        except Exception:
            pass
        event.accept()
