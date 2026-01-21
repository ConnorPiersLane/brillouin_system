import logging
import threading
import time

from brillouin_system.eye_tracker.eye_position.coordinates import RigCoord
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config_gui import EyeTrackerConfigDialog
from brillouin_system.eye_tracker.eye_tracker_results import get_eye_tracker_results, EyeTrackerResults
from brillouin_system.guis.human_interface.eye_tracker_controller import EyeTrackerController
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig
from brillouin_system.scan_managers.scanning_config.scanning_config_gui import \
    AxialScanningConfigDialog
from brillouin_system.logging_utils import logging_setup

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

# Must be set before QApplication is constructed:
QtCore.QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
# QtCore.QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


import sys
import pickle
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget, TextItem

from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGroupBox, QLabel, QLineEdit,
    QFileDialog, QPushButton, QHBoxLayout, QFormLayout, QVBoxLayout, QCheckBox, QComboBox, QListWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import QtCore

from brillouin_system.logging_utils.qt_log_handler import QtLogBridge, QtTextEditHandler
from brillouin_system.logging_utils.logging_setup import start_logging, install_crash_hooks, get_logger, \
    shutdown_logging, logging_fmt_gui, enable_console_fallback

log = get_logger(__name__)

from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.calibration.config.calibration_config_gui import CalibrationConfigDialog
from brillouin_system.devices.cameras.allied.allied_config.allied_config_dialog import AlliedConfigDialog
from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from brillouin_system.devices.cameras.andor.andor_frame.andor_config_dialog import AndorConfigDialog
from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterface, \
    ZaberHumanInterfaceDummy
from brillouin_system.guis.human_interface.hi_backend import HiBackend
from brillouin_system.guis.human_interface.hi_signaller import HiSignaller
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
# from brillouin_system.devices.cameras.mako.allied_vision_camera import AlliedVisionCamera
from brillouin_system.devices.microwave_device import MicrowaveDummy, Microwave
from brillouin_system.devices.shutter_device import ShutterManagerDummy, ShutterManager
from brillouin_system.guis.data_analyzer.show_axial_scan import AxialScanViewer
from brillouin_system.my_dataclasses.background_image import BackgroundImage
from brillouin_system.my_dataclasses.human_interface_measurements import RequestAxialStepScan, AxialScan, \
    RequestAxialContScan
from brillouin_system.calibration.calibration import render_calibration_to_pixmap, \
    CalibrationImageDialog, CalibrationData, CalibrationCalculator

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLensDummy, ZaberEyeLens

###
# Add other guis
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import dataclass_to_hdf5_native_dict, save_dict_to_hdf5
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FittingConfigs
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config_gui import FindPeaksConfigDialog


use_backend_dummy = True
# Eye Tracking
include_eye_tracking = True
use_eye_tracker_dummy = False

# put this near your imports (top of file)
class NotifyingViewBox(pg.ViewBox):
    userScaled = QtCore.pyqtSignal()
    def wheelEvent(self, ev, axis=None):
        super().wheelEvent(ev, axis)
        self.userScaled.emit()
    def mouseDragEvent(self, ev, axis=None):
        super().mouseDragEvent(ev, axis)
        if ev.isFinish():
            self.userScaled.emit()


def create_backend(use_dummy: bool) -> HiBackend:
    # Testing
    if use_dummy:
        backend = HiBackend(
            camera=DummyCamera(),
            shutter_manager=ShutterManagerDummy('human_interface'),
            microwave=MicrowaveDummy(),
            # zaber_eye_lens=ZaberEyeLensDummy(),
            # zaber_hi=ZaberHumanInterfaceDummy(),
            zaber_eye_lens=ZaberEyeLens(),
            zaber_hi=ZaberHumanInterface(),
            is_sample_illumination_continuous=True
        )

    else:
        backend = HiBackend(
            camera=IxonUltra(
                index = 0,
                temperature = "off",
                fan_mode = "full",
                x_start = 40, x_end  = 120,
                y_start= 300, y_end  = 315,
                vbin= 1, hbin  = 1,
                verbose = True,
                advanced_gain_option=False
            ),
            shutter_manager=ShutterManager('human_interface'),
            microwave=Microwave(),
            zaber_eye_lens=ZaberEyeLens(),
            zaber_hi=ZaberHumanInterface(),
            is_sample_illumination_continuous=True
        )
    return backend

class HiFrontend(QWidget):

    # Signals Outgoing
    apply_camera_settings_requested = pyqtSignal(dict)
    toggle_camera_shutter_requested = pyqtSignal()
    emit_camera_settings_requested = pyqtSignal()
    start_live_requested = pyqtSignal()
    stop_live_requested = pyqtSignal()
    update_microwave_freq_requested = pyqtSignal(float)
    toggle_illumination_requested = pyqtSignal()
    toggle_bg_subtraction_requested = pyqtSignal()
    snap_requested = pyqtSignal()
    toggle_reference_mode_requested = pyqtSignal()
    acquire_background_requested = pyqtSignal()

    move_zaber_eye_lens_requested = pyqtSignal(float)
    move_zaber_stage_x_requested = pyqtSignal(float)
    move_zaber_stage_y_requested = pyqtSignal(float)
    move_zaber_stage_z_requested = pyqtSignal(float)

    run_calibration_requested = pyqtSignal()
    take_axial_step_scan_requested = pyqtSignal(object)
    take_axial_cont_scan_requested = pyqtSignal(object)
    shutdown_requested = pyqtSignal()
    get_calibration_results_requested = pyqtSignal()
    toggle_do_live_fitting_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    update_andor_config_requested = pyqtSignal(object)
    close_all_shutters_requested = pyqtSignal()
    update_fitting_configs_requested = pyqtSignal(FittingConfigs)
    request_axial_scan_data = pyqtSignal(int)
    update_scanning_config_requested = pyqtSignal(object)
    take_bg_value_reflection_plane_request = pyqtSignal()
    find_reflection_plane_request = pyqtSignal()

    # Saving Signals
    save_all_axial_scans_requested = pyqtSignal()
    save_selected_axial_scans_requested = pyqtSignal(list)
    remove_selected_axial_scans_requested = pyqtSignal(list)

    # Eye Tracking Signals
    set_et_allied_configs = pyqtSignal(object, object)  # left_cfg, right_cfg
    request_eye_shutdown = pyqtSignal()
    set_et_config = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        # Attribute
        self.history_data = deque(maxlen=100)


        self.laser_focus_position: RigCoord | None = None
        self.last_eye_tracker_results: EyeTrackerResults | None = None
        self._last_eye_update_monotonic = 0.0
        self._andor_exposure_time: float | None = None

        # --- Create log_view early so logging can safely use it ---
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(120)

        # --- Logging to GUI (Qt handler) ---
        self.log_bridge = QtLogBridge()
        self.log_bridge.message.connect(self._append_log_line)

        self.qt_handler = QtTextEditHandler(self.log_bridge)
        # Use the same format as your file logs for consistency
        self.qt_handler.setFormatter(logging_fmt_gui)

        lg = get_logger()
        if self.qt_handler not in lg.handlers:
            lg.addHandler(self.qt_handler)
        lg.setLevel(logging.INFO)


        self.setWindowTitle("Brillouin Viewer (Live)")

        self.brillouin_signaller = HiSignaller(manager=create_backend(use_backend_dummy))
        self.brillouin_signaller_thread = QThread()
        self.brillouin_signaller.moveToThread(self.brillouin_signaller_thread)

        # Sending signals
        self.apply_camera_settings_requested.connect(self.brillouin_signaller.apply_camera_settings)
        self.emit_camera_settings_requested.connect(self.brillouin_signaller.emit_camera_settings)
        self.toggle_camera_shutter_requested.connect(self.brillouin_signaller.toggle_camera_shutter)
        self.start_live_requested.connect(self.brillouin_signaller.start_live_view)
        self.stop_live_requested.connect(self.brillouin_signaller.stop_live_view)
        self.update_microwave_freq_requested.connect(self.brillouin_signaller.set_microwave_frequency)
        self.toggle_illumination_requested.connect(self.brillouin_signaller.toggle_illumination_mode)
        self.toggle_bg_subtraction_requested.connect(self.brillouin_signaller.toggle_background_subtraction)
        self.snap_requested.connect(self.brillouin_signaller.snap_and_fit)
        self.toggle_reference_mode_requested.connect(self.brillouin_signaller.toggle_reference_mode)
        self.acquire_background_requested.connect(self.brillouin_signaller.acquire_background_image)

        self.move_zaber_eye_lens_requested.connect(self.brillouin_signaller.move_zaber_eye_lens_relative)
        self.move_zaber_stage_x_requested.connect(self.brillouin_signaller.move_zaber_stage_x_relative)
        self.move_zaber_stage_y_requested.connect(self.brillouin_signaller.move_zaber_stage_y_relative)
        self.move_zaber_stage_z_requested.connect(self.brillouin_signaller.move_zaber_stage_z_relative)

        self.run_calibration_requested.connect(self.brillouin_signaller.run_calibration)
        self.take_axial_step_scan_requested.connect(self.brillouin_signaller.take_axial_step_scan)
        self.take_axial_cont_scan_requested.connect(self.brillouin_signaller.take_axial_cont_scan)
        self.shutdown_requested.connect(self.brillouin_signaller.close)
        self.get_calibration_results_requested.connect(self.brillouin_signaller.get_calibration_results)
        self.toggle_do_live_fitting_requested.connect(self.brillouin_signaller.toggle_do_live_fitting)
        self.cancel_requested.connect(self.brillouin_signaller.cancel_operations)
        self.update_andor_config_requested.connect(self.brillouin_signaller.update_andor_config_settings)
        self.close_all_shutters_requested.connect(self.brillouin_signaller.close_all_shutters)
        self.update_fitting_configs_requested.connect(self.brillouin_signaller.update_fitting_configs)
        self.request_axial_scan_data.connect(self.brillouin_signaller.handle_request_axial_scan_data)
        self.update_scanning_config_requested.connect(self.brillouin_signaller.update_scanning_config)
        self.find_reflection_plane_request.connect(self.brillouin_signaller.delegate_find_reflection_plane)

        # Receiving signals
        self.brillouin_signaller.calibration_finished.connect(self.calibration_finished)
        self.brillouin_signaller.background_subtraction_state.connect(self.update_bg_subtraction)
        self.brillouin_signaller.background_available_state.connect(self.handle_is_bg_available)
        self.brillouin_signaller.illumination_mode_state.connect(self.update_illumination_ui)
        self.brillouin_signaller.reference_mode_state.connect(self.update_reference_ui)
        self.brillouin_signaller.camera_settings_ready.connect(self.populate_camera_ui)
        self.brillouin_signaller.camera_shutter_state_changed.connect(self.update_camera_shutter_button)
        self.brillouin_signaller.new_andor_display_ready.connect(self._on_andor_mailbox, Qt.QueuedConnection)

        self.brillouin_signaller.zaber_lens_position_updated.connect(self.update_zaber_lens_position)
        self.brillouin_signaller.zaber_stage_positions_updated.connect(self.update_stage_positions)

        self.brillouin_signaller.microwave_frequency_updated.connect(self.update_ref_freq_input)

        self.brillouin_signaller.calibration_result_ready.connect(self.handle_requested_calibration)
        self.brillouin_signaller.do_live_fitting_state.connect(self.update_do_live_fitting_checkbox)

        self.brillouin_signaller.update_system_state_in_frontend.connect(self.update_system_state_label)
        self.brillouin_signaller.send_update_stored_axial_scans.connect(self.receive_axial_scan_list)
        self.brillouin_signaller.axial_scan_data_ready.connect(self.handle_received_axial_scan_data)
        self.brillouin_signaller.send_axial_scans_to_save.connect(self.save_axial_scan_list_to_file)
        self.brillouin_signaller.send_message_to_frontend.connect(self.message_handler)
        self.brillouin_signaller.close_event_finished.connect(self._finalize_close)

        # Saving Signals
        self.save_all_axial_scans_requested.connect(self.brillouin_signaller.save_all_axial_scans)
        self.save_selected_axial_scans_requested.connect(self.brillouin_signaller.save_multiple_axial_scans)
        self.remove_selected_axial_scans_requested.connect(self.brillouin_signaller.remove_selected_axial_scans)





        # Connect signals BEFORE starting the thread
        self.brillouin_signaller_thread.started.connect(self.run_gui)

        # Start the thread after all connections
        self.brillouin_signaller_thread.start()

        # Build UI (reuses self.log_view created above)
        self.init_ui()

        self.update_gui()

        # --- Eye tracker thread & controller ---
        if include_eye_tracking:
            self.start_eye_tracker()

    def start_eye_tracker(self):
        self.eye_thread = QThread(self)
        self.eye_ctrl = EyeTrackerController(use_dummy=use_eye_tracker_dummy)  # or False
        self.eye_ctrl.moveToThread(self.eye_thread)

        # start/stop/shutdown
        self.eye_thread.started.connect(self.eye_ctrl.start)

        # frames into GUI
        self.eye_ctrl.frames_ready.connect(self.on_eye_frames_ready)

        # Sending signals
        self.set_et_allied_configs.connect(self.eye_ctrl.proxy.set_allied_configs)
        self.set_et_config.connect(self.eye_ctrl.send_config)
        self.request_eye_shutdown.connect(self.eye_ctrl.shutdown)

        self.eye_thread.start()

    def _append_log_line(self, line: str):
        self.log_view.append(line)

    def init_ui(self):
        outer_layout = QHBoxLayout()
        self.setLayout(outer_layout)

        # LEFT COLUMN: Controls
        left_column_layout = QVBoxLayout()
        left_column_layout.addWidget(self.create_control_group())
        left_column_layout.addWidget(self.create_andor_camera_group())
        left_column_layout.addWidget(self.create_fitting_group())
        left_column_layout.addWidget(self.create_reference_group())
        left_column_layout.addWidget(self.create_background_group())
        left_column_layout.addWidget(self.create_illumination_group())
        left_column_layout.addWidget(self.create_allied_vision_group())
        left_column_layout.addWidget(self.create_axial_scans_group())
        left_column_layout.addWidget(self.create_show_scan_results())
        left_column_layout.addStretch()
        outer_layout.addLayout(left_column_layout, 0)

        # MID COLUMN: Plots + log view
        middle_column_layout = QVBoxLayout()
        middle_column_layout.addWidget(self.create_andor_display_group())

        zaber_movement_and_scan_layout = QHBoxLayout()
        first_column = QVBoxLayout()

        first_column.addWidget(self.create_zaber_manual_movement_group())
        first_column.addWidget(self.create_axial_scan_settings_group())
        zaber_movement_and_scan_layout.addLayout(first_column)

        axial_column = QVBoxLayout()
        axial_column.addWidget(self.create_take_axial_scan_group())
        axial_column.addWidget(self.create_axial_scan_cont_group())

        zaber_movement_and_scan_layout.addLayout(axial_column)
        middle_column_layout.addLayout(zaber_movement_and_scan_layout)

        # log_view was created in __init__; just add it here
        # middle_column_layout.addWidget(self.log_view)

        outer_layout.addLayout(middle_column_layout, 1)

        # FAR RIGHT COLUMN: Eye Tracking
        right_column_layout = QVBoxLayout()
        if include_eye_tracking:
            right_column_layout.addWidget(self.create_eye_tracking_group())
            right_column_layout.addStretch()
            outer_layout.addLayout(right_column_layout, 0)

        right_column_layout.addWidget(self.create_log_group())
        right_column_layout.addStretch()
        outer_layout.addLayout(right_column_layout, 0)

    # ---------------- UI Sections ---------------- #

    def create_control_group(self):
        group = QGroupBox("Control")
        layout = QHBoxLayout()

        stop_btn = QPushButton("STOP")
        cancel_btn = QPushButton("Cancel")
        restart_btn = QPushButton("Restart")

        self.state_label = QLabel("● IDLE")
        self.state_label.setStyleSheet("color: gray; font-weight: bold")

        stop_btn.clicked.connect(self.on_stop_clicked)
        cancel_btn.clicked.connect(self.on_cancel_event_clicked)
        restart_btn.clicked.connect(self.on_restart_clicked)

        layout.addWidget(stop_btn)
        layout.addWidget(cancel_btn)
        layout.addWidget(restart_btn)
        layout.addWidget(self.state_label)
        group.setLayout(layout)
        return group

    def create_background_group(self):
        self.bg_label_off = QLabel("● No BG Subtraction")
        self.bg_label_on = QLabel("○ With BG Subtraction")

        self.bg_label_off.setStyleSheet("color: green; font-weight: bold")
        self.bg_label_on.setStyleSheet("color: gray")

        self.btn_take_bg = QPushButton("Take BG")
        self.btn_take_bg.clicked.connect(self.take_background_image)

        self.toggle_bg_btn = QPushButton("Switch")
        self.toggle_bg_btn.clicked.connect(self.toggle_background_subtraction)

        self.btn_save_bg = QPushButton("Save BG")
        self.btn_save_bg.clicked.connect(self.save_background_image)

        # Create horizontal layout for the three buttons
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_take_bg)
        btn_row.addWidget(self.toggle_bg_btn)
        btn_row.addWidget(self.btn_save_bg)

        layout = QVBoxLayout()
        layout.addWidget(self.bg_label_off)
        layout.addWidget(self.bg_label_on)
        layout.addSpacing(5)
        layout.addLayout(btn_row)  # Add button row as one horizontal layout

        group = QGroupBox("Background")
        group.setLayout(layout)
        return group

    def create_andor_camera_group(self):
        self.exposure_input = QLineEdit()
        self.exposure_input.setValidator(QDoubleValidator(0.001, 60.0, 3))

        self.gain_input = QLineEdit()
        self.gain_input.setValidator(QIntValidator(0, 1000))

        self.config_camera_btn = QPushButton("Settings")
        self.config_camera_btn.clicked.connect(self.on_andor_configs_clicked)

        self.toggle_camera_shutter_btn = QPushButton("Close")
        self.toggle_camera_shutter_btn.clicked.connect(self.toggle_camera_shutter_requested.emit)

        self.apply_camera_btn = QPushButton("Apply")
        self.apply_camera_btn.clicked.connect(self.apply_camera_settings)

        # Horizontal layout for the buttons
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.config_camera_btn)
        btn_row.addWidget(self.toggle_camera_shutter_btn)
        btn_row.addWidget(self.apply_camera_btn)

        # Main layout
        layout = QFormLayout()
        layout.addRow("Exp. Time (s):", self.exposure_input)
        layout.addRow("Gain:", self.gain_input)
        layout.addRow(btn_row)

        group = QGroupBox("Andor Camera")
        group.setLayout(layout)

        return group

    def create_fitting_group(self):
        self.fitting_config_btn = QPushButton("Config")
        self.fitting_config_btn.clicked.connect(self.on_fitting_configs_clicked)

        self.do_live_fitting_checkbox = QCheckBox("Do Live Sample Fitting")
        self.do_live_fitting_checkbox.stateChanged.connect(self.on_do_live_fitting_toggled)

        # Horizontal layout for button + checkbox
        row_layout = QHBoxLayout()
        row_layout.addWidget(self.fitting_config_btn)
        row_layout.addWidget(self.do_live_fitting_checkbox)

        # Vertical layout for the group box
        layout = QVBoxLayout()
        layout.addLayout(row_layout)

        group = QGroupBox("Fitting")
        group.setLayout(layout)

        return group

    def create_illumination_group(self):
        self.illum_label_cont = QLabel("● Cont.")
        self.illum_label_pulse = QLabel("○ Pulsed")

        self.illum_label_cont.setStyleSheet("color: gray")
        self.illum_label_pulse.setStyleSheet("color: gray")

        self.toggle_illum_btn = QPushButton("Switch")
        self.toggle_illum_btn.clicked.connect(self.toggle_illumination)

        self.snap_once_btn = QPushButton("Take Snap")
        self.snap_once_btn.clicked.connect(self.run_one_gui_update)

        # Row 1: Continuous label and toggle button
        row1 = QHBoxLayout()
        row1.addWidget(self.illum_label_cont)
        row1.addWidget(self.toggle_illum_btn)
        row1.addStretch()

        # Row 2: Pulsed label and snap button
        row2 = QHBoxLayout()
        row2.addWidget(self.illum_label_pulse)
        row2.addWidget(self.snap_once_btn)
        row2.addStretch()

        # Combine rows vertically
        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)

        group = QGroupBox("Illumination")
        group.setLayout(layout)
        return group

    def create_reference_group(self):
        # Mode labels
        self.calib_label_meas = QLabel("● Meas.")
        self.calib_label_calib = QLabel("○ Ref.")
        self.calib_label_meas.setStyleSheet("color: green; font-weight: bold")
        self.calib_label_calib.setStyleSheet("color: gray")

        self.toggle_calib_btn = QPushButton("Switch")
        self.toggle_calib_btn.setFixedWidth(60)
        self.toggle_calib_btn.clicked.connect(self.toggle_reference_mode)

        mode_column = QVBoxLayout()
        mode_column.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        mode_column.addWidget(self.calib_label_meas, alignment=Qt.AlignHCenter)
        mode_column.addWidget(self.calib_label_calib, alignment=Qt.AlignHCenter)
        mode_column.addSpacing(4)
        mode_column.addWidget(self.toggle_calib_btn, alignment=Qt.AlignHCenter)

        # Frequency row
        self.ref_freq_input = QLineEdit()
        self.ref_freq_input.setFixedWidth(50)
        self.ref_freq_input.setValidator(QDoubleValidator(0.0, 100.0, 4))
        self.ref_freq_input.setText("5.5")

        self.set_ref_btn = QPushButton("Set")
        self.set_ref_btn.setFixedWidth(40)
        self.set_ref_btn.clicked.connect(self.set_reference_freq)

        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Ref. Freq. (GHz):"))
        freq_row.addWidget(self.ref_freq_input)
        freq_row.addWidget(self.set_ref_btn)
        freq_row.addStretch()

        # Config + Cal. all
        self.config_ref_btn = QPushButton("Config")
        self.config_ref_btn.setFixedWidth(60)
        self.config_ref_btn.clicked.connect(self.on_reference_configs_clicked)

        self.calibrate_btn = QPushButton("Cal. all")
        self.calibrate_btn.setFixedWidth(70)
        self.calibrate_btn.clicked.connect(self.run_calibration)

        config_row = QHBoxLayout()
        config_row.addWidget(self.config_ref_btn)
        config_row.addWidget(self.calibrate_btn)
        config_row.addStretch()

        # Show + Save
        self.show_calib_btn = QPushButton("Show")
        self.show_calib_btn.setFixedWidth(60)
        self.show_calib_btn.clicked.connect(self.show_calibration_results)
        self.show_calib_btn.setEnabled(False)

        self.save_calib_btn = QPushButton("Save")
        self.save_calib_btn.setFixedWidth(60)
        self.save_calib_btn.clicked.connect(self.save_calibration_results)
        self.save_calib_btn.setEnabled(False)

        save_row = QHBoxLayout()
        save_row.addWidget(self.show_calib_btn)
        save_row.addWidget(self.save_calib_btn)
        save_row.addStretch()

        # Right side layout (all rows stacked)
        right_column = QVBoxLayout()
        right_column.setAlignment(Qt.AlignTop)
        right_column.addLayout(freq_row)
        right_column.addLayout(config_row)
        right_column.addLayout(save_row)

        # Combine left and right in fixed container layout
        content_layout = QHBoxLayout()
        content_layout.setAlignment(Qt.AlignLeft)
        content_layout.addLayout(mode_column)
        content_layout.addSpacing(15)
        content_layout.addLayout(right_column)

        # Wrap in main vertical layout to avoid stretching
        outer_layout = QVBoxLayout()
        outer_layout.setAlignment(Qt.AlignTop)
        outer_layout.addLayout(content_layout)

        group = QGroupBox("Reference")
        group.setLayout(outer_layout)
        return group

    def create_allied_vision_group(self):
        group = QGroupBox("Eye Tracker")

        # Buttons row for Allied Vision Cameras
        btn_row = QHBoxLayout()
        self.btn_allied_left = QPushButton("Left")
        self.btn_allied_right = QPushButton("Right")
        self.btn_eye_tracking = QPushButton("Config")

        btn_row.addWidget(self.btn_allied_left)
        btn_row.addWidget(self.btn_allied_right)
        btn_row.addWidget(self.btn_eye_tracking)
        btn_row.addStretch()

        btn_row2 = QHBoxLayout()
        self.btn_restart_eye = QPushButton("ReStart")
        self.btn_close_eye = QPushButton("Shutdown")

        btn_row2.addWidget(self.btn_restart_eye)
        btn_row2.addWidget(self.btn_close_eye)
        btn_row2.addStretch()

        # put the row into a real layout and set it on the group
        v = QVBoxLayout()
        v.addLayout(btn_row)
        v.addLayout(btn_row2)
        group.setLayout(v)

        # wire up
        self.btn_allied_left.clicked.connect(self.open_allied_left_dialog)
        self.btn_allied_right.clicked.connect(self.open_allied_right_dialog)
        self.btn_eye_tracking.clicked.connect(self.open_eye_tracker_config_dialog)
        self.btn_restart_eye.clicked.connect(self.on_restart_eye_clicked)
        self.btn_close_eye.clicked.connect(self.shutdown_eye_tracker)

        return group

    def create_axial_scan_settings_group(self):
        group = QGroupBox("Axial Scan Settings and Reflection Plane Finding")
        layout = QFormLayout()

        self.axial_settings_btn = QPushButton("Settings")
        self.axial_settings_btn.clicked.connect(self.open_axial_scan_settings_dialog)

        self.find_reflection_plane_btn = QPushButton("Find Reflection Plane")
        self.find_reflection_plane_btn.clicked.connect(self.find_reflection_plane)


        btn_row = QHBoxLayout()
        btn_row.addWidget(self.axial_settings_btn)
        # btn_row.addWidget(self.take_bg_value_btn)
        btn_row.addWidget(self.find_reflection_plane_btn)

        btn_row.addStretch()

        layout.addRow(btn_row)

        group.setLayout(layout)
        return group

    def create_take_axial_scan_group(self):
        group = QGroupBox("Scan Axial Steps")
        layout = QFormLayout()

        # --- Axial scan (step mode) controls ---
        self.axial_id_input = QLineEdit()
        self.axial_id_input.setFixedWidth(80)
        self.axial_num_input = QLineEdit("10")
        self.axial_num_input.setFixedWidth(80)
        self.axial_num_input.textChanged.connect(self.update_axial_step_distance)
        self.axial_step_input = QLineEdit("10")
        self.axial_step_input.setFixedWidth(80)
        self.axial_step_input.textChanged.connect(self.update_axial_step_distance)

        self.axial_steps_scanned_dist_label = QLabel("100.00 µm")
        self.axial_steps_scanned_dist_label.setFixedWidth(80)


        layout.addRow("ID:", self.axial_id_input)
        layout.addRow("Num Meas:", self.axial_num_input)
        layout.addRow("Step Size (µm):", self.axial_step_input)
        layout.addRow("Scanned Dist (µm):", self.axial_steps_scanned_dist_label)

        self.axial_btn = QPushButton("Scan")
        self.axial_btn.clicked.connect(lambda: self.take_axial_step_scan(find_reflection_plane=False))

        self.axial_btn2 = QPushButton("Find -> Scan")
        self.axial_btn2.clicked.connect(lambda: self.take_axial_step_scan(find_reflection_plane=True))

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.axial_btn)
        btn_row.addWidget(self.axial_btn2)
        btn_row.addStretch()

        layout.addRow(btn_row)

        group.setLayout(layout)
        return group

    def create_axial_scan_cont_group(self):
        """
        Continuous axial scan widget.

        Uses Andor exposure time ("Exp. Time (s)") and speed (µm/s) to compute:
            Scanned Dist (µm) = speed * exposure_time

        User still specifies:
            - ID
            - Max Dist (µm)
        """
        group = QGroupBox("Axial Scan Cont.")
        layout = QFormLayout()

        # ID (similar to step scan)
        self.axial_cont_id_input = QLineEdit()
        self.axial_cont_id_input.setFixedWidth(80)

        # Speed (µm/s)
        self.axial_cont_speed_input = QLineEdit("500")
        self.axial_cont_speed_input.setFixedWidth(80)
        self.axial_cont_speed_input.textChanged.connect(self.update_axial_cont_distance)

        # Computed scanned distance (read-only)
        self.axial_cont_scanned_dist_label = QLabel("0.00 µm")
        self.axial_cont_scanned_dist_label.setFixedWidth(80)


        layout.addRow("ID:", self.axial_cont_id_input)
        layout.addRow("Speed (µm/s):", self.axial_cont_speed_input)
        layout.addRow("Scanned Dist (µm):", self.axial_cont_scanned_dist_label)

        # Button (no backend wiring yet)
        self.axial_cont_btn = QPushButton("Scan")
        self.axial_cont_btn.clicked.connect(lambda: self.take_axial_cont_scan(find_reflection_plane=False))

        self.axial_cont_btn2 = QPushButton("Find -> Scan")  # <-- name/text as you like
        self.axial_cont_btn2.clicked.connect(lambda: self.take_axial_cont_scan(find_reflection_plane=True))

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.axial_cont_btn)
        btn_row.addWidget(self.axial_cont_btn2)
        btn_row.addStretch()

        layout.addRow(btn_row)

        group.setLayout(layout)

        # Initialize the scanned distance from current exposure & speed

        return group

    def update_axial_step_distance(self):
        """
        Compute Scanned Dist (µm) = #steps-1 () * distance (um)
        and update the label in the Axial Scan Cont. group.
        """
        # If the continuous scan widgets aren't created yet, just ignore
        try:
            # Speed (µm/s) from Axial Scan Cont.
            N = self.axial_num_input.text().strip()
            N = int(N)

            step = self.axial_step_input.text().strip()
            step = float(step)

            dist_um = N * step
            self.axial_steps_scanned_dist_label.setText(f"{dist_um:.2f} µm")
        except Exception:
            # If parsing fails, show a neutral placeholder
            self.axial_steps_scanned_dist_label.setText("—")

    def update_axial_cont_distance(self):
        """
        Compute Scanned Dist (µm) = speed (µm/s) * exposure_time (s, from Andor)
        and update the label in the Axial Scan Cont. group.
        """
        # If the continuous scan widgets aren't created yet, just ignore
        if not hasattr(self, "axial_cont_scanned_dist_label"):
            return

        try:
            # Speed (µm/s) from Axial Scan Cont.
            speed_str = self.axial_cont_speed_input.text().strip()
            speed = float(speed_str) if speed_str else 0.0

            # Exposure time (s) from Andor camera group
            exposure_s = self._andor_exposure_time if self._andor_exposure_time else 0.0

            dist_um = speed * exposure_s
            self.axial_cont_scanned_dist_label.setText(f"{dist_um:.2f} µm")
        except Exception:
            # If parsing fails, show a neutral placeholder
            self.axial_cont_scanned_dist_label.setText("—")


    def create_log_group(self):
        group = QGroupBox("Log Output")
        layout = QVBoxLayout()
        layout.addWidget(self.log_view)
        group.setLayout(layout)
        return group

    # --- Eye Tracking UI ---

    def create_eye_tracking_group(self):
        group = QGroupBox("Eye Tracking")

        self.eye_glw = GraphicsLayoutWidget()
        self.eye_glw.ci.setContentsMargins(4, 4, 4, 4)
        self.eye_glw.ci.setSpacing(4)

        self.eye_vb = []
        self.eye_img = []

        # NEW: store bottom-row plots
        self.eye_bottom_plots = []

        for i in range(6):
            if i in (2, 4):
                self.eye_glw.ci.nextRow()

            row = i // 2
            col = i % 2

            # Top 2 rows (0..3): keep your existing "image in a ViewBox"
            if i < 4:
                vb = pg.ViewBox(lockAspect=True, enableMenu=False)
                vb.invertY(True)
                img = pg.ImageItem(autoDownsample=True)
                vb.addItem(img)
                vb.setBorder((80, 80, 80))

                self.eye_glw.ci.addItem(vb, row=row, col=col)
                self.eye_vb.append(vb)
                self.eye_img.append(img)

            # Bottom row (4..5): make them true 1D plots with axes
            else:
                p = self.eye_glw.ci.addPlot(row=row, col=col)
                p.showGrid(x=True, y=False, alpha=0.2)
                p.setLabel("bottom", "Distance (mm)")
                p.hideAxis("left")
                p.setYRange(-1, 1, padding=0.0)
                p.setXRange(-3.0, 10.0, padding=0.0)
                p.getViewBox().setBorder((80, 80, 80))
                p.setMenuEnabled(False)

                self.eye_bottom_plots.append(p)

        # Make last row smaller (row index 2)
        grid = self.eye_glw.ci.layout
        grid.setRowStretchFactor(0, 6)
        grid.setRowStretchFactor(1, 6)
        grid.setRowStretchFactor(2, 1)

        self._init_laser_position_map()  # laser map uses self.eye_vb[2] :contentReference[oaicite:1]{index=1}
        self._init_cornea_distance_plot()  # NEW

        layout = QVBoxLayout()
        layout.addWidget(self.eye_glw)
        group.setLayout(layout)
        group.setMinimumWidth(800)
        group.setMinimumHeight(800)
        return group

    def _init_cornea_distance_plot(self):
        """
        Bottom-left plot (index 0 in self.eye_bottom_plots):
          - Laser dot at x=0
          - Grey band indicating cornea region [dc, dc+thickness]
        """
        if not getattr(self, "eye_bottom_plots", None):
            return

        self.cornea_plot = self.eye_bottom_plots[0]  # bottom-left
        # If you want bottom-right too, use self.eye_bottom_plots[1]

        # Laser marker at x=0
        self.laser_1d_dot = pg.ScatterPlotItem(
            [0.0], [0.0],
            size=10,
            brush=pg.mkBrush("r"),
            pen=pg.mkPen("r")
        )
        self.cornea_plot.addItem(self.laser_1d_dot)

        # Cornea band (defaults for now)
        self.cornea_thickness_mm = 0.5
        dc_mm = 2.0
        self.cornea_band = pg.LinearRegionItem(
            values=(dc_mm, dc_mm + self.cornea_thickness_mm),
            orientation=pg.LinearRegionItem.Vertical,
            movable=False,
            brush=pg.mkBrush(150, 150, 150, 80),
            pen=pg.mkPen(150, 150, 150, 200),
        )
        self.cornea_plot.addItem(self.cornea_band)

    def create_show_scan_results(self):
        """
        Simple widget: just a scan selection dropdown and a Show button.
        """
        # ComboBox for selecting axial scan
        self.shift_scan_combo = QComboBox()
        self.shift_scan_combo.setMinimumWidth(160)

        # Show button
        self.shift_show_btn = QPushButton("Show")
        self.shift_show_btn.setFixedWidth(60)
        self.shift_show_btn.clicked.connect(self.on_show_axial_scan_clicked)

        # Horizontal layout
        row = QHBoxLayout()
        row.addWidget(self.shift_scan_combo)
        row.addWidget(self.shift_show_btn)
        row.addStretch()

        # Final layout
        layout = QVBoxLayout()
        layout.addLayout(row)
        layout.addStretch()

        group = QGroupBox("Show Axial Scans")
        group.setLayout(layout)
        return group

    def create_axial_scans_group(self):
        group = QGroupBox("Axial Scans")

        # List widget to display axial scan entries
        self.axial_scans_list = QListWidget()
        self.axial_scans_list.setSelectionMode(QListWidget.MultiSelection)

        # Buttons
        self.save_all_scans_btn = QPushButton("Save All")
        self.save_selected_scan_btn = QPushButton("Save Selected")
        self.remove_selected_scan_btn = QPushButton("Remove Selected")

        self.save_all_scans_btn.clicked.connect(self.save_all_axial_scans)
        self.save_selected_scan_btn.clicked.connect(self.save_selected_axial_scan)
        self.remove_selected_scan_btn.clicked.connect(self.remove_selected_axial_scan)

        # Horizontal layout for buttons
        button_row = QHBoxLayout()
        button_row.addWidget(self.save_all_scans_btn)
        button_row.addWidget(self.save_selected_scan_btn)
        button_row.addWidget(self.remove_selected_scan_btn)

        # Main vertical layout
        layout = QVBoxLayout()
        layout.addWidget(self.axial_scans_list)
        layout.addLayout(button_row)

        group.setLayout(layout)
        return group

    # --- inside HiFrontend ---
    def create_andor_display_group(self):
        # White theme + fast options
        pg.setConfigOptions(
            useOpenGL=False,
            antialias=True,
            imageAxisOrder='row-major',
            background='w',
            foreground='k'
        )

        group = QGroupBox("Andor Frame and Fitting")
        lay = QVBoxLayout(group)

        self.glw = GraphicsLayoutWidget()
        lay.addWidget(self.glw)

        ci = self.glw.ci
        ci.setSpacing(12)
        ci.setContentsMargins(8, 8, 8, 8)

        # -------- Row 1: Live image --------
        self.vb_img = self.glw.addViewBox(lockAspect=True, enableMenu=False)
        self.img_item = pg.ImageItem(autoDownsample=True)
        self.vb_img.addItem(self.img_item)
        self.vb_img.setBorder((170, 170, 170))

        # -------- Row 2: Spectrum + fit --------
        self.glw.nextRow()
        self.spec_vb = NotifyingViewBox()
        self.spec_plot = self.glw.addPlot(
            viewBox=self.spec_vb,
            labels={'left': 'Intensity', 'bottom': 'Pixel (X)'}
        )
        self.spec_plot.setMenuEnabled(False)
        self.spec_plot.setClipToView(False)
        self.spec_plot.enableAutoRange(x=True, y=True)
        self.spec_plot.setTitle("Spectrum + Fit", color=(40, 40, 40))
        self.spec_plot.getViewBox().setBorder((170, 170, 170))
        self.spec_vb.userScaled.connect(self._on_spec_user_scaled)

        # Fit: red dashed line (no symbols)
        self.fit_curve = pg.PlotDataItem(
            pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.DashLine)
        )
        self.fit_curve.setZValue(0)
        self.spec_plot.addItem(self.fit_curve)

        # All measurement samples: black dots (no connecting line)
        self.spec_points = pg.PlotDataItem(
            pen=None,
            symbol='o',
            symbolSize=4,
            symbolPen=pg.mkPen('k'),
            symbolBrush='k'
        )
        self.spec_points.setZValue(1)
        self.spec_plot.addItem(self.spec_points)

        # Points used for fitting: red dots overlay
        self.mask_points = pg.PlotDataItem(
            pen=None,
            symbol='o',
            symbolSize=4,
            symbolPen=pg.mkPen('r'),
            symbolBrush='r'
        )
        self.mask_points.setZValue(2)
        self.spec_plot.addItem(self.mask_points)

        # -------- Row 3: History --------
        self.glw.nextRow()
        self.hist_vb = NotifyingViewBox()
        self.hist_plot = self.glw.addPlot(
            viewBox=self.hist_vb,
            labels={'left': 'GHz', 'bottom': 'Frame'}
        )
        self.hist_plot.setMenuEnabled(False)
        self.hist_plot.setClipToView(False)
        self.hist_plot.enableAutoRange(x=True, y=True)
        self.hist_plot.setTitle("Shift History", color=(40, 40, 40))
        self.hist_plot.getViewBox().setBorder((170, 170, 170))
        self.hist_vb.userScaled.connect(self._on_hist_user_scaled)

        self.hist_curve = self.hist_plot.plot([], [], pen=pg.mkPen('k', width=1))

        # flags/caches
        self._spec_init_done = False
        self._spec_user_zoomed = False
        self._hist_user_zoomed = False
        self.HIST_WINDOW = 200  # trailing window size for scrolling

        return group

    # --- inside HiFrontend ---
    def _on_spec_user_scaled(self):
        self._spec_user_zoomed = True

    def _on_hist_user_scaled(self):
        self._hist_user_zoomed = True

    def create_zaber_manual_movement_group(self):
        group = QGroupBox("Manually Move Zabers")
        layout = QFormLayout()

        # Helper to build each row
        def make_movement_row(label_widget, step_input, *buttons):
            row = QHBoxLayout()
            row.addWidget(step_input)
            for btn in buttons:
                row.addWidget(btn)
            row.addStretch()
            return label_widget, row

        # LENS AXIS
        self.lens_step_input = QLineEdit("100")
        self.lens_step_input.setFixedWidth(60)

        self.lens_back_btn = QPushButton("← Back")
        self.lens_forward_btn = QPushButton("→ Forward")

        self.lens_pos_display = QLabel("Lens 0.00 µm")  # stored label
        layout.addRow(
            *make_movement_row(self.lens_pos_display, self.lens_step_input, self.lens_back_btn, self.lens_forward_btn))

        # Z STAGE AXIS
        self.z_step_input = QLineEdit("100")
        self.z_step_input.setFixedWidth(60)

        self.z_back_btn = QPushButton("← Back")
        self.z_forward_btn = QPushButton("→ Forward")

        self.z_pos_display = QLabel("Stage Z 0.00 µm")  # stored label
        layout.addRow(*make_movement_row(self.z_pos_display, self.z_step_input, self.z_back_btn, self.z_forward_btn))

        # X STAGE AXIS
        self.x_step_input = QLineEdit("100")
        self.x_step_input.setFixedWidth(60)

        self.x_left_btn = QPushButton("← Left")
        self.x_right_btn = QPushButton("→ Right")

        self.x_pos_display = QLabel("Stage X 0.00 µm")  # stored label
        layout.addRow(*make_movement_row(self.x_pos_display, self.x_step_input, self.x_left_btn, self.x_right_btn))

        # Y STAGE AXIS
        self.y_step_input = QLineEdit("100")
        self.y_step_input.setFixedWidth(60)

        self.y_up_btn = QPushButton("↑ Up")
        self.y_down_btn = QPushButton("↓ Down")

        self.y_pos_display = QLabel("Stage Y 0.00 µm")  # stored label
        layout.addRow(*make_movement_row(self.y_pos_display, self.y_step_input, self.y_up_btn, self.y_down_btn))


        # Connect lens movement buttons
        self.lens_back_btn.clicked.connect(lambda: self.move_zaber_lens_by(-1))
        self.lens_forward_btn.clicked.connect(lambda: self.move_zaber_lens_by(+1))

        # Stage Z
        self.z_back_btn.clicked.connect(
            lambda: self.move_zaber_stage_z_requested.emit(-float(self.z_step_input.text())))
        self.z_forward_btn.clicked.connect(
            lambda: self.move_zaber_stage_z_requested.emit(+float(self.z_step_input.text())))

        # Stage X
        self.x_left_btn.clicked.connect(
            lambda: self.move_zaber_stage_x_requested.emit(+float(self.x_step_input.text())))
        self.x_right_btn.clicked.connect(
            lambda: self.move_zaber_stage_x_requested.emit(-float(self.x_step_input.text())))

        # Stage Y
        self.y_up_btn.clicked.connect(lambda: self.move_zaber_stage_y_requested.emit(+float(self.y_step_input.text())))
        self.y_down_btn.clicked.connect(
            lambda: self.move_zaber_stage_y_requested.emit(-float(self.y_step_input.text())))


        # ----------------------------
        # NEW: Move laser to (R, phi) using eye tracking (ABSOLUTE target)
        # ----------------------------
        self.xy_r_input = QLineEdit("0.0")
        self.xy_r_input.setFixedWidth(60)
        self.xy_r_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 4))  # mm

        self.xy_phi_input = QLineEdit("0.0")
        self.xy_phi_input.setFixedWidth(60)
        self.xy_phi_input.setValidator(QDoubleValidator(-3600.0, 3600.0, 3))  # deg

        self.xy_move_btn = QPushButton("Move XY")
        self.xy_move_btn.setFixedWidth(70)
        self.xy_move_btn.clicked.connect(self.on_move_xy_polar_clicked)

        xy_row = QHBoxLayout()
        xy_row.addWidget(QLabel("R [mm]:"))
        xy_row.addWidget(self.xy_r_input)
        xy_row.addSpacing(6)
        xy_row.addWidget(QLabel("phi [deg]:"))
        xy_row.addWidget(self.xy_phi_input)
        xy_row.addSpacing(6)
        xy_row.addWidget(self.xy_move_btn)
        xy_row.addStretch()

        layout.addRow(xy_row)

        # ----------------------------
        # NEW: Move Z to reach Δc target (uses current Δc from eye tracking)
        # ----------------------------
        self.dc_input = QLineEdit("2.0")
        self.dc_input.setFixedWidth(60)
        self.dc_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 4))  # mm

        self.dc_move_btn = QPushButton("Move Z")
        self.dc_move_btn.setFixedWidth(70)
        self.dc_move_btn.clicked.connect(self.on_move_z_by_dc_clicked)

        dc_row = QHBoxLayout()
        dc_row.addWidget(QLabel("Δc [mm]:"))
        dc_row.addWidget(self.dc_input)
        dc_row.addSpacing(6)
        dc_row.addWidget(self.dc_move_btn)
        dc_row.addStretch()

        layout.addRow(dc_row)

        group.setLayout(layout)
        return group

    # ---------------- GUI Update Loop ---------------- #
    def _on_andor_mailbox(self):
        display = self.brillouin_signaller.fetch_latest_andor_display()
        if display is None:
            return
        self._display_result_fast(display)  # your fast pyqtgraph updater

    def on_stop_clicked(self):
        log.info("[Brillouin Viewer] STOP clicked.")

        # --- EMERGENCY: close shutters in a separate thread immediately ---
        try:
            threading.Thread(
                target=self.brillouin_signaller.backend.shutter_manager.close_all,
                daemon=True
            ).start()
        except Exception as e:
            log.exception("[STOP] Failed to spawn emergency shutter close thread: %s", e)

        # Existing logic: cancel operations + clean stop of live view
        self.brillouin_signaller.cancel_operations()
        self.close_all_shutters_requested.emit()
        self.stop_live_requested.emit()

    def on_cancel_event_clicked(self):
        log.info("[BrillouinViewer] Cancel button clicked.")
        # self.cancel_requested.emit()
        self.brillouin_signaller.cancel_operations()

    def on_restart_clicked(self):
        log.info("[Brillouin Viewer] Restart clicked.")
        self.stop_live_requested.emit()
        QApplication.processEvents()
        self.start_live_requested.emit()

    def update_gui(self):
        # Update the guis
        self.brillouin_signaller.update_gui()

    def save_all_axial_scans(self):
        self.save_all_axial_scans_requested.emit()

    def save_selected_axial_scan(self):
        selected_items = self.axial_scans_list.selectedItems()
        if not selected_items:
            log.warning("No scans selected.")
            return

        indices = [int(item.text().split(" - ")[0]) for item in selected_items]
        self.save_selected_axial_scans_requested.emit(indices)

    def save_axial_scan_list_to_file(self, scans: list):
        from PyQt5.QtWidgets import QFileDialog
        from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
            dataclass_to_hdf5_native_dict, save_dict_to_hdf5
        )

        if not scans:
            log.info("[Save] No axial scans to save.")
            return

        base_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Axial Scans (base name)",
            filter="All Files (*)"
        )
        if not base_path:
            return

        try:
            # Save as Pickle
            pkl_path = base_path if base_path.endswith(".pkl") else base_path + ".pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(scans, f)
            log.info(f"[✓] Pickle saved to: {pkl_path}")

            # Save as HDF5
            h5_path = base_path if base_path.endswith(".h5") else base_path + ".h5"
            native_dict = dataclass_to_hdf5_native_dict(scans)
            save_dict_to_hdf5(h5_path, native_dict)
            log.info(f"[✓] HDF5 saved to: {h5_path}")

        except Exception as e:
            log.exception(f"[Error] Failed to save axial scans: {e}")

    def move_zaber_lens_by(self, direction: int):
        try:
            step = float(self.lens_step_input.text())
            self.move_zaber_eye_lens_requested.emit(direction * step)
        except ValueError:
            log.exception("[Error] Invalid lens step size input.")

    def update_stage_positions(self, x: float, y: float, z: float):
        self.x_pos_display.setText(f"X {x:.2f} µm")
        self.y_pos_display.setText(f"Y {y:.2f} µm")
        self.z_pos_display.setText(f"Z {z:.2f} µm")

    def remove_selected_axial_scan(self):
        selected_items = self.axial_scans_list.selectedItems()
        if not selected_items:
            log.warning("[Warning] No scan selected.")
            return
        indices = [int(item.text().split(" - ")[0]) for item in selected_items]
        self.remove_selected_axial_scans_requested.emit(indices)

    # ---------------- Signal Handles ---------------- #




    def update_system_state_label(self, state):
        if state.name == "IDLE":
            self.state_label.setText("● IDLE")
            self.state_label.setStyleSheet("color: gray; font-weight: bold")
        elif state.name == "BUSY":
            self.state_label.setText("● BUSY")
            self.state_label.setStyleSheet("color: orange; font-weight: bold")
        elif state.name == "FREERUNNING":
            self.state_label.setText("● LIVE")
            self.state_label.setStyleSheet("color: green; font-weight: bold")

    def on_andor_configs_clicked(self):
        dialog = AndorConfigDialog(andor_update_config=self.update_andor_config_settings, parent=self)
        dialog.exec_()

    def update_andor_config_settings(self, andor_config: AndorConfig):
        self.update_andor_config_requested.emit(andor_config)

    def request_scanning_config_file_update(self, scanning_config: ScanningConfig):
        self.update_scanning_config_requested.emit(scanning_config)

    def update_fitting_configs(self, fitting_configs: FittingConfigs):
        self.update_fitting_configs_requested.emit(fitting_configs)

    def on_fitting_configs_clicked(self):
        dialog = FindPeaksConfigDialog(on_apply=self.update_fitting_configs, parent=self)
        dialog.exec_()

    def on_reference_configs_clicked(self):
        dialog = CalibrationConfigDialog(self)
        dialog.exec_()


    def update_bg_subtraction(self, enabled: bool):
        if enabled:
            self.bg_label_on.setText("● With BG Subtraction")
            self.bg_label_on.setStyleSheet("color: green; font-weight: bold")
            self.bg_label_off.setText("○ No BG Subtraction")
            self.bg_label_off.setStyleSheet("color: gray")
        else:
            self.bg_label_on.setText("○ With BG Subtraction")
            self.bg_label_on.setStyleSheet("color: gray")
            self.bg_label_off.setText("● No BG Subtraction")
            self.bg_label_off.setStyleSheet("color: green; font-weight: bold")

    def handle_is_bg_available(self, available: bool):
        self.toggle_bg_btn.setEnabled(available)
        self.btn_save_bg.setEnabled(available)


    def on_do_live_fitting_toggled(self, state: int):
        self.toggle_do_live_fitting_requested.emit()

    def update_do_live_fitting_checkbox(self, state: bool):
        self.do_live_fitting_checkbox.setChecked(state)

    def update_illumination_ui(self, is_cont: bool):
        if is_cont:
            self.illum_label_cont.setText("● Cont.")
            self.illum_label_cont.setStyleSheet("color: green; font-weight: bold")
            self.illum_label_pulse.setText("○ Pulsed")
            self.illum_label_pulse.setStyleSheet("color: gray")
            self.snap_once_btn.setEnabled(False)
            self.run_gui()  # restart live view
        else:
            self.illum_label_cont.setText("○ Cont.")
            self.illum_label_cont.setStyleSheet("color: gray")
            self.illum_label_pulse.setText("● Pulsed")
            self.illum_label_pulse.setStyleSheet("color: green; font-weight: bold")
            self.snap_once_btn.setEnabled(True)



    # ---------------- Toggle ---------------- #
    def toggle_background_subtraction(self):
        self.toggle_bg_subtraction_requested.emit()

    def toggle_illumination(self):
        self.toggle_illumination_requested.emit()




    def update_zaber_lens_position(self, pos: float):
        self.lens_pos_display.setText(f"Lens {pos:.2f} µm")
        self.laser_focus_position = RigCoord(x=0, y=0, z=pos/1000)

    # ---------------- GUI Update Loop ---------------- #


    def run_gui(self):
        self.start_live_requested.emit()


    def run_one_gui_update(self):
        self.snap_requested.emit()

    # --- inside HiFrontend ---
    def _display_result_fast(self, dr):
        # -------- 1) Image: ALWAYS min-max per frame --------
        frame = np.ascontiguousarray(dr.frame)
        if frame.dtype == np.float64:
            frame = frame.astype(np.float32, copy=False)

        fmin = float(np.nanmin(frame)) if frame.size else 0.0
        fmax = float(np.nanmax(frame)) if frame.size else 1.0
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
            fmin, fmax = 0.0, 1.0

        self.img_item.setImage(frame, autoLevels=False, levels=(fmin, fmax))

        # -------- 2) Spectrum + Fit --------
        x = np.asarray(getattr(dr, "x_pixels", []))
        y = np.asarray(getattr(dr, "sline", []))

        self.spec_points.setData(x, y)

        if getattr(dr, "is_fitting_available", False):
            xf = np.asarray(getattr(dr, "x_fit_refined", []))
            yf = np.asarray(getattr(dr, "y_fit_refined", []))
            self.fit_curve.setData(xf, yf)
        else:
            self.fit_curve.setData([], [])

        mask = getattr(dr, "mask_for_fitting", None)
        if mask is not None and x.size and y.size:
            m = np.asarray(mask, dtype=bool)
            n = min(m.shape[0], x.shape[0], y.shape[0])
            self.mask_points.setData(x[:n][m[:n]], y[:n][m[:n]])
            self.mask_points.setVisible(True)
        else:
            self.mask_points.setData([], [])
            self.mask_points.setVisible(False)

        # one-time autorange (don’t fight user after they zoom)
        if not self._spec_init_done and x.size and y.size:
            vb = self.spec_plot.getViewBox()
            vb.setRange(
                xRange=(float(x.min()), float(x.max())),
                yRange=(float(y.min()), float(y.max())),
                padding=0.05
            )
            self._spec_init_done = True
        elif not self._spec_user_zoomed:
            # allow gentle auto-follow on Y until the user interacts
            self.spec_plot.enableAutoRange(y=True)

        # -------- 3) History --------
        val = getattr(dr, "freq_shift_ghz", None)
        if val is not None:
            self.history_data.append(float(val))
            N = len(self.history_data)

            x_hist = np.arange(N, dtype=float)
            y_hist = np.fromiter(self.history_data, dtype=float)
            self.hist_curve.setData(x_hist, y_hist)

            vb = self.hist_plot.getViewBox()
            if not self._hist_user_zoomed:
                if N <= self.HIST_WINDOW:
                    vb.setXRange(0, max(10, N), padding=0)
                else:
                    left = N - self.HIST_WINDOW
                    right = N
                    vb.setXRange(left, right, padding=0)
                self.hist_plot.enableAutoRange(y=True)

    # ---------------- Handlers ---------------- #

    def update_camera_shutter_button(self, is_open: bool):
        text = "Close" if is_open else "Open"
        self.toggle_camera_shutter_btn.setText(text)

    def populate_camera_ui(self, settings: dict):
        self._andor_exposure_time = settings["exposure"]
        self.update_axial_cont_distance()
        self.exposure_input.setText(str(self._andor_exposure_time))
        self.gain_input.setText(str(settings["gain"]))


    def apply_camera_settings(self):
        try:
            exposure = round(float(self.exposure_input.text()), ndigits=4)
            gain = int(self.gain_input.text())

            settings = {
                "exposure": exposure,
                "gain": gain,
            }

            self.apply_camera_settings_requested.emit(settings)
            log.info("[Brillouin Viewer] Sent new camera settings to worker.")

            self.emit_camera_settings_requested.emit()

        except Exception as e:
            log.exception(f"[Brillouin Viewer] Failed to apply camera settings: {e}")


    def toggle_reference_mode(self):
        self.toggle_reference_mode_requested.emit()

    def update_reference_ui(self, is_reference_mode: bool):
        if is_reference_mode:
            self.calib_label_meas.setText("○ Meas.")
            self.calib_label_meas.setStyleSheet("color: gray")
            self.calib_label_calib.setText("● Ref.")
            self.calib_label_calib.setStyleSheet("color: green; font-weight: bold")
        else:
            self.calib_label_meas.setText("● Meas.")
            self.calib_label_meas.setStyleSheet("color: green; font-weight: bold")
            self.calib_label_calib.setText("○ Ref.")
            self.calib_label_calib.setStyleSheet("color: gray")


    def update_ref_freq_input(self, freq: float):
        self.ref_freq_input.setText(f"{freq:.3f}")

    # -------------- Functions --------------


    def save_background_image(self):
        def receive_data(data: BackgroundImage):
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Background Image", filter="All Files (*)"
            )
            if not path:
                return

            try:
                # Save as Pickle
                pkl_path = path if path.endswith(".pkl") else path + ".pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(data, f)
                log.info(f"[✓] Background image saved to: {pkl_path}")

                # Save as HDF5
                h5_path = path if path.endswith(".h5") else path + ".h5"
                native_dict = dataclass_to_hdf5_native_dict(data)
                save_dict_to_hdf5(h5_path, native_dict)
                log.info(f"[✓] Background image saved as HDF5 to: {h5_path}")

            except Exception as e:
                log.exception(f"[Brillouin Viewer] [Error] Failed to save background data: {e}")

            finally:
                self.brillouin_signaller.background_data_ready.disconnect(receive_data)

        self.brillouin_signaller.background_data_ready.connect(receive_data)
        self.brillouin_signaller.emit_background_data()



    def set_reference_freq(self):
        try:
            freq = float(self.ref_freq_input.text())
            self.update_microwave_freq_requested.emit(freq)
        except ValueError:
            log.exception("[Brillouin Viewer] [Reference] Invalid frequency input.")

    def take_background_image(self):
        self.acquire_background_requested.emit()

    def receive_axial_scan_list(self, scan_list: list):

        # Update QListWidget
        self.axial_scans_list.clear()
        self.axial_scans_list.addItems(scan_list)

        self.shift_scan_combo.clear()
        self.shift_scan_combo.addItems(scan_list)

    def on_show_axial_scan_clicked(self):
        selected_scan = self.shift_scan_combo.currentText()
        if not selected_scan:
            log.info("[Brillouin Viewer] No axial available.")
            return

        i = int(selected_scan.split(" - ")[0])

        self.request_axial_scan_data.emit(i)

    def handle_received_axial_scan_data(self, scan_data: AxialScan):
        try:
            self.axial_viewer = AxialScanViewer(scan_data)
            self.axial_viewer.show()
        except Exception as e:
            log.exception(f"[AxialScanViewer] Failed to show data: {e}")

    def run_calibration(self):
        self.run_calibration_requested.emit()

    def show_calibration_results(self):
        self._show_cali = True
        self._save_cali = False
        self.get_calibration_results_requested.emit()

    def save_calibration_results(self):
        self._show_cali = False
        self._save_cali = True
        self.get_calibration_results_requested.emit()

    def calibration_finished(self):
        self.show_calib_btn.setEnabled(True)
        self.save_calib_btn.setEnabled(True)
        log.info(f"[Brillouin Viewer] Calibration available")

    def handle_requested_calibration(self, received_cali: tuple[CalibrationData, CalibrationCalculator]):
        cali_data = received_cali[0]
        cali_calculator = received_cali[1]

        if self._show_cali:
            try:
                pixmap = render_calibration_to_pixmap(
                    cali_data, cali_calculator, calibration_config.get().reference
                )
                dialog = CalibrationImageDialog(pixmap, parent=self)
                dialog.exec_()
                log.info("[Brillouin Viewer] Calibration plot displayed.")
            except Exception as e:
                log.exception(f"[Brillouin Viewer] Failed to plot calibration: {e}")

        elif self._save_cali:
            if cali_data is None:
                log.info("[Brillouin Viewer] Failed to save data, no data available")
                return

            base_path, _ = QFileDialog.getSaveFileName(
                self, "Save Calibration Data", filter="All Files (*)"
            )
            if not base_path:
                return

            try:
                # Save Pickle
                pkl_path = base_path if base_path.endswith(".pkl") else base_path + ".pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(cali_data, f)
                log.info(f"[✓] Calibration data saved to {pkl_path}")

                # Save HDF5
                h5_path = base_path if base_path.endswith(".h5") else base_path + ".h5"
                hdf5_dict = dataclass_to_hdf5_native_dict(cali_data)
                save_dict_to_hdf5(h5_path, hdf5_dict)
                log.info(f"[✓] Calibration data saved as HDF5 to {h5_path}")

            except Exception as e:
                log.exception(f"[Error] Failed to save calibration data: {e}")


        self._show_cali = False
        self._save_cali = True

    def take_axial_step_scan(self, find_reflection_plane: bool = False):
        try:
            id_str = self.axial_id_input.text().strip()
            n_meas = int(self.axial_num_input.text())
            step = float(self.axial_step_input.text())

            # Log info
            log.info(
                f"[Brillouin Viewer] Axial Scan Request | ID: {id_str}, N: {n_meas}, Step: {step} µm")


            request = RequestAxialStepScan(
                id=id_str,
                n_measurements=n_meas,
                step_size_um=step,
                find_reflection_plane=find_reflection_plane,
                eye_tracker_results=self.last_eye_tracker_results,
            )

            self.take_axial_step_scan_requested.emit(request)

        except Exception as e:
            log.exception(f"[Brillouin Viewer] Failed to initiate axial scan: {e}")

    def take_axial_cont_scan(self, find_reflection_plane: bool = False):
        try:
            id_str = self.axial_cont_id_input.text().strip()
            speed = float(self.axial_cont_speed_input.text())

            # Log info
            log.info(
                f"[Brillouin Viewer] Axial Continuous Scan Request | ID: {id_str} with {speed}µm/s")


            request = RequestAxialContScan(
                id=id_str,
                speed_um_s=speed,
                find_reflection_plane=find_reflection_plane,
                eye_tracker_results=self.last_eye_tracker_results,
            )

            self.take_axial_cont_scan_requested.emit(request)

        except Exception as e:
            log.exception(f"[Brillouin Viewer] Failed to initiate axial scan: {e}")

    def clear_measurements(self):
        self._stored_measurements.clear()
        self.measurement_series_label.setText("Stored Series: 0")
        log.info("[Brillouin Viewer] Cleared all stored measurement series.")

    # -- Open Dialog --
    def message_handler(self, title: str, message: str):
        QMessageBox.information(self, title, message)

    # ---- Allied config dialogs ----

    def open_allied_left_dialog(self):
        dlg = AlliedConfigDialog("left", self._apply_allied_left, parent=self)
        dlg.exec_()

    def open_allied_right_dialog(self):
        dlg = AlliedConfigDialog("right", self._apply_allied_right, parent=self)
        dlg.exec_()

    def _apply_allied_left(self, cfg_obj):
        """
        Called by AlliedConfigDialog on Apply for LEFT.
        `cfg_obj` is the updated AlliedConfig for the left camera.
        We emit both left & right configs to the eye-tracker proxy.
        """
        # Push both configs into the eye tracker proxy via our signal
        self.set_et_allied_configs.emit(cfg_obj, None)

    def _apply_allied_right(self, cfg_obj):
        """
        Called by AlliedConfigDialog on Apply for RIGHT.
        """

        self.set_et_allied_configs.emit(None, cfg_obj)

    def open_eye_tracker_config_dialog(self):
        """
        Open the EyeTrackerConfigDialog and on Apply send the config to the
        eye-tracker worker via EyeTrackerProxy.set_et_config.
        """

        def _on_apply(cfg: EyeTrackerConfig):
            """
            Called by EyeTrackerConfigDialog.apply() after updating the
            global ThreadSafeConfig. `cfg` is an EyeTrackerConfig instance.
            """
            try:
                # Emit our signal, which is connected to proxy.set_et_config
                self.set_et_config.emit(cfg)
                log.info("[EyeTracker] Sent new eye tracker config to proxy.")
            except Exception as e:
                log.exception(f"[EyeTracker] Failed to send config to proxy: {e}")

        dlg = EyeTrackerConfigDialog(
            on_apply=_on_apply,
            parent=self,
        )
        dlg.exec_()

    def open_axial_scan_settings_dialog(self):

        def _on_apply(cfg: ScanningConfig):
            try:
                # Emit our signal, which is connected to proxy.set_et_config
                self.update_scanning_config_requested.emit(cfg)
                log.info("[Frontend] Sent new axial scan settings.")
            except Exception as e:
                log.exception(f"[EyeTracker] Failed to send new axial scan settings: {e}")

        dlg = AxialScanningConfigDialog(
            on_apply=_on_apply,
            parent=self,
        )
        dlg.exec_()

    def _init_laser_position_map(self):
        """
        Initializes the position map display in eye_vb[2].
        Includes:
            - Concentric circles (radius 1..4)
            - Radial lines at 0°, 45°, 90°, 135°
            - Moving laser point
            - Text display (x, y, Δz) in upper-right corner
        """
        vb = self.eye_vb[2]

        # -------------------------------
        # Configure ViewBox
        # -------------------------------
        vb.setAspectLocked(True)  # Keep it square
        vb.setRange(xRange=(-4, 4), yRange=(-4, 4))
        vb.invertY(False)  # make y-axis normal (upward is +y)
        # If you want "y from 4 down to -4", then:
        # vb.invertY(True)

        # -------------------------------
        # Draw concentric circles
        # -------------------------------
        angles = np.linspace(0, 2 * np.pi, 720)
        for r in range(1, 5):
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            circle = pg.PlotDataItem(
                x, y,
                pen=pg.mkPen(150, 150, 150)  # light gray
            )
            vb.addItem(circle)

        # -------------------------------
        # Radial lines: 0°, 45°, 90°, 135°
        # -------------------------------
        for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315, ]:
            theta = np.deg2rad(angle_deg)
            x = [1 * np.cos(theta), 4 * np.cos(theta)]
            y = [1 * np.sin(theta), 4 * np.sin(theta)]

            line = pg.PlotDataItem(
                x, y,
                pen=pg.mkPen(150, 150, 150, style=QtCore.Qt.DashLine)
            )
            vb.addItem(line)

        # -------------------------------
        # Moving laser dot
        # -------------------------------
        self.laser_point = pg.ScatterPlotItem(
            [0.0], [0.0],
            size=12,
            brush=pg.mkBrush('r'),
            pen=pg.mkPen('r')
        )
        vb.addItem(self.laser_point)

        # -------------------------------
        # Text display (top-right)
        # -------------------------------
        self.pos_text = TextItem(
            text="x=\ny=\nz=\nΔc=",
            color=(0, 0, 0),
            anchor=(1, 0)  # right-top anchor
        )
        vb.addItem(self.pos_text)

        # Place it at the exact upper-right of the defined range
        # If invertY(False): top = +4
        # If invertY(True):  top = -4
        top_y = 4 if not vb.yInverted() else -4

        self.pos_text.setPos(4, top_y)

    def update_laser_position_cartesian(self, x: float, y: float):
        if not (np.isfinite(x) and np.isfinite(y)):
            self.clear_laser_position()
            return
        self.laser_point.setData([float(x)], [float(y)])
        self.laser_point.setVisible(True)

    def clear_laser_position(self):
        self.laser_point.setVisible(False)

    def update_laser_position_text_eye_tracker(self, x=None, y=None, z=None, dc=None):
        """
        Update the x, y, Δz display with fixed width fields.
        If a value is None, show blank padded fields.
        """

        def fmt(v):
            return f"{v:6.3f}" if v is not None else " " * 6

        tx = fmt(x)
        ty = fmt(y)
        tz = fmt(z)
        tdc = fmt(dc)

        self.pos_text.setText(
            f"x = {tx}\n"
            f"y = {ty}\n"
            f"z = {tz}\n"
            f"Δc = {tdc}"
        )


    @QtCore.pyqtSlot(object, object, dict)
    def on_eye_frames_ready(self, left, right, meta):
        """
        (left, right, meta)
          left     : np.ndarray (H, W, 3), uint8
          right    : np.ndarray (H, W, 3), uint8
          meta     : Dict: {"ts": last["ts"], "idx": last["idx"], "pupil3D": pupil3D}
        """
        # left/right/rendered: np.ndarray(H, W, 3), uint8

        # pyqtgraph's ImageItem can handle 3-channel images, but a safe option is
        # to convert to grayscale for now:

        self._last_eye_update_monotonic = time.monotonic()
        self.eye_img[0].setImage(left, autoLevels=True)
        self.eye_img[1].setImage(right, autoLevels=True)

        self.last_eye_tracker_results = get_eye_tracker_results(
            left=left, right=right, meta=meta, laser_focus_position=self.laser_focus_position
        )
        laser_position = self.last_eye_tracker_results.laser_position
        if laser_position is not None:
            self.update_laser_position_cartesian(x=laser_position[0], y=laser_position[1])
            self.update_laser_position_text_eye_tracker(
                x=laser_position[0],
                y=laser_position[1],
                z=laser_position[2],
                dc=self.last_eye_tracker_results.delta_laser_corner
            )

            self._update_cornea_band(self.last_eye_tracker_results.delta_laser_corner)

        else:
            self.clear_laser_position()
            self.update_laser_position_text_eye_tracker(None, None, None, None)

            # IMPORTANT: also hide cornea when laser_position missing
            self._update_cornea_band(None)

    def _wait_for_eye_result(self, timeout_s: float = 1, max_age_s: float = 0.5) -> EyeTrackerResults | None:
        """
        Wait (briefly) for a recent EyeTrackerResults with a valid laser_position.
        max_age_s: how recent the result must be.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            r = self.last_eye_tracker_results
            if r is not None and r.laser_position is not None:
                age = time.monotonic() - self._last_eye_update_monotonic
                if age <= max_age_s:
                    return r
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
        return None

    def _update_cornea_band(self, dc_mm):
        # If plot isn't initialized yet, do nothing
        if not hasattr(self, "cornea_band"):
            return

        if dc_mm is None:
            self.cornea_band.setVisible(False)
            return

        self.cornea_band.setRegion((dc_mm, dc_mm + self.cornea_thickness_mm))
        self.cornea_band.setVisible(True)


    def on_move_xy_polar_clicked(self):
        """
        Move stage so the laser ends up at (R, phi) in laser coordinates.
        R [mm], phi [deg] describe the ABSOLUTE desired laser position.
        """
        try:
            res = self._wait_for_eye_result(timeout_s=0.8, max_age_s=0.3)
            if res is None:
                log.warning("[Zaber XY Polar] No recent eye-tracker result.")
                return

            x_laser = float(res.laser_position[0])
            y_laser = float(res.laser_position[1])

            r_mm = float(self.xy_r_input.text())
            phi_deg = float(self.xy_phi_input.text())

            phi = np.deg2rad(phi_deg)
            x_target = r_mm * np.cos(phi)
            y_target = r_mm * np.sin(phi)

            dx_laser = x_target - x_laser
            dy_laser = y_target - y_laser

            dx_um = dx_laser * 1000.0
            dy_um = dy_laser * 1000.0

            log.info(
                "[Zaber XY Polar] laser=(%.3f, %.3f) -> target=(%.3f, %.3f) mm | Δ=(%.3f, %.3f) mm",
                x_laser, y_laser, x_target, y_target, dx_laser, dy_laser
            )

            # SIGN NOTE:
            # If moving the stage +X makes the laser move -X,
            # change emit(dx_um) -> emit(-dx_um) (same for Y).
            self.move_zaber_stage_x_requested.emit(-dx_um)
            self.move_zaber_stage_y_requested.emit(dy_um)

        except Exception as e:
            log.exception(f"[Zaber XY Polar] Failed: {e}")

    def on_move_z_by_dc_clicked(self):
        """
        Move Z so that Δc reaches the user-specified value (in mm).
        Uses current Δc from eye tracking (delta_laser_corner, assumed µm).
        """
        try:
            res = self._wait_for_eye_result(timeout_s=0.8, max_age_s=0.3)
            if res is None:
                log.warning("[Zaber Z Δc] No recent eye-tracker result (or laser_position None).")
                return

            dc_current_mm = res.delta_laser_corner
            if dc_current_mm is None:
                log.warning("[Zaber Z Δc] delta_laser_corner is None; cannot compute Z move.")
                return

            dc_target_mm = float(self.dc_input.text())

            dz_mm = dc_target_mm - dc_current_mm
            dz_um = dz_mm * 1000.0

            log.info("[Zaber Z Δc] dc_current=%.3fmm target=%.3fmm -> dz=%.3fmm",
                     dc_current_mm, dc_target_mm, dz_mm)

            # Same sign caveat as XY: if moving +Z increases Δc or decreases it depends on your geometry.
            self.move_zaber_stage_z_requested.emit(-dz_um)

        except Exception as e:
            log.exception(f"[Zaber Z Δc] Invalid input or update error: {e}")

    # ---- Fitting button (placeholder) ----

    def shutdown_eye_tracker(self):
        self.request_eye_shutdown.emit()
        time.sleep(5)
        self.eye_thread.quit()
        self.eye_thread.wait()

    def on_restart_eye_clicked(self):
        # Close old eye tracker
        self.shutdown_eye_tracker()
        # Create fresh controller + thread
        self.start_eye_tracker()


    def find_reflection_plane(self):
        self.find_reflection_plane_request.emit()


    def closeEvent(self, event):
        print("GUI shutdown initiated...")
        event.ignore()

        if include_eye_tracking:
            self.shutdown_eye_tracker()

        self.stop_live_requested.emit()
        self.shutdown_requested.emit()  # no sleep

    def _finalize_close(self):
        print("Backend shutdown complete. Closing GUI...")
        self.brillouin_signaller_thread.quit()
        self.brillouin_signaller_thread.wait(3000)

        QApplication.quit()


def main():
    # Set rounding policy before constructing QApplication (Qt ≥ 5.14)
    try:
        if hasattr(QtWidgets.QApplication, "setHighDpiScaleFactorRoundingPolicy"):
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor
            )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        * { font-size: 8pt; }
    """)
    viewer = HiFrontend()
    viewer.show()
    exit_code = app.exec_()
    sys.exit(exit_code)



if __name__ == "__main__":
    # IMPORTANT: On Windows, spawn the writer process ONLY here.
    start_logging()
    install_crash_hooks()

    # For debugging:
    # enable_console_fallback(force=True)

    main()
