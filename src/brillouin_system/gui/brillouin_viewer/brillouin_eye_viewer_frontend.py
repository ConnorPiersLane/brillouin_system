
import sys
import pickle
import numpy as np

from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGroupBox, QLabel, QLineEdit,
    QFileDialog, QPushButton, QHBoxLayout, QFormLayout, QVBoxLayout, QCheckBox, QComboBox, QListWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.calibration.config.calibration_config_gui import CalibrationConfigDialog


from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from brillouin_system.devices.cameras.andor.andor_frame.andor_config_dialog import AndorConfigDialog
from brillouin_system.gui.brillouin_viewer.brillouin_backend import BrillouinBackend
from brillouin_system.gui.brillouin_viewer.brillouin_signaller import BrillouinSignaller
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
# from brillouin_system.devices.cameras.mako.allied_vision_camera import AlliedVisionCamera
from brillouin_system.devices.microwave_device import MicrowaveDummy
from brillouin_system.devices.shutter_device import ShutterManagerDummy
from brillouin_system.gui.helpers.show_axial_scan import AxialScanViewer

from brillouin_system.my_dataclasses.background_image import BackgroundImage
from brillouin_system.my_dataclasses.human_interface_measurements import RequestAxialScan, AxialScan

from brillouin_system.calibration.calibration import render_calibration_to_pixmap, \
    CalibrationImageDialog, CalibrationData, CalibrationCalculator
from brillouin_system.my_dataclasses.display_results import DisplayResults
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLensDummy



###
# Add other guis

from brillouin_system.saving_and_loading.safe_and_load_hdf5 import dataclass_to_hdf5_native_dict, save_dict_to_hdf5
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FittingConfigs
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config_gui import FindPeaksConfigDialog

## Testing
brillouin_manager = BrillouinBackend(
    system_type='human_interface',
        camera=DummyCamera(),
    shutter_manager=ShutterManagerDummy('human_interface'),
    microwave=MicrowaveDummy(),
    zaber_eye_lens=ZaberEyeLensDummy(),
    is_sample_illumination_continuous=True
)


# # # Real
# brillouin_manager = BrillouinBackend(
#         camera=IxonUltra(
#             index = 0,
#             temperature = -80, #"off"
#             fan_mode = "full",
#             x_start = 40, x_end  = 120,
#             y_start= 300, y_end  = 315,
#             vbin= 1, hbin  = 1,
#             verbose = True,
#             advanced_gain_option=False
#         ),
#     shutter_manager=ShutterManager('human_interface'),
#     microwave=Microwave(),
#     zaber=ZaberLinearController(),
#     is_sample_illumination_continuous=True
# )


class BrillouinEyeViewerFrontend(QWidget):

    # Signals Outgoing
    gui_ready = pyqtSignal()
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
    run_calibration_requested = pyqtSignal()
    take_axial_scan_requested = pyqtSignal(object)
    shutdown_requested = pyqtSignal()
    get_calibration_results_requested = pyqtSignal()
    toggle_do_live_fitting_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    update_andor_config_requested = pyqtSignal(object)
    close_all_shutters_requested = pyqtSignal()
    update_fitting_configs_requested = pyqtSignal(FittingConfigs)
    request_axial_scan_data = pyqtSignal(int)

    # Saving Signals
    save_all_axial_scans_requested = pyqtSignal()
    save_selected_axial_scans_requested = pyqtSignal(list)
    remove_selected_axial_scans_requested = pyqtSignal(list)



    def __init__(self):
        super().__init__()

        # State control
        self._show_live = True

        self.setWindowTitle("Brillouin Viewer (Live)")

        self.brillouin_signaller = BrillouinSignaller(manager=brillouin_manager)
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
        self.run_calibration_requested.connect(self.brillouin_signaller.run_calibration)
        self.take_axial_scan_requested.connect(self.brillouin_signaller.take_axial_scan)
        self.gui_ready.connect(self.brillouin_signaller.on_gui_ready)
        self.shutdown_requested.connect(self.brillouin_signaller.close)
        self.get_calibration_results_requested.connect(self.brillouin_signaller.get_calibration_results)
        self.toggle_do_live_fitting_requested.connect(self.brillouin_signaller.toggle_do_live_fitting)
        self.cancel_requested.connect(self.brillouin_signaller.cancel_operations)
        self.update_andor_config_requested.connect(self.brillouin_signaller.update_andor_config_settings)
        self.close_all_shutters_requested.connect(self.brillouin_signaller.close_all_shutters)
        self.update_fitting_configs_requested.connect(self.brillouin_signaller.update_fitting_configs)
        self.request_axial_scan_data.connect(self.brillouin_signaller.handle_request_axial_scan_data)

        # Receiving signals
        self.brillouin_signaller.calibration_finished.connect(self.calibration_finished)
        self.brillouin_signaller.background_subtraction_state.connect(self.update_bg_subtraction)
        self.brillouin_signaller.background_available_state.connect(self.handle_is_bg_available)
        self.brillouin_signaller.illumination_mode_state.connect(self.update_illumination_ui)
        self.brillouin_signaller.reference_mode_state.connect(self.update_reference_ui)
        self.brillouin_signaller.camera_settings_ready.connect(self.populate_camera_ui)
        self.brillouin_signaller.camera_shutter_state_changed.connect(self.update_camera_shutter_button)
        self.brillouin_signaller.frame_and_fit_ready.connect(self.display_result, Qt.QueuedConnection)
        self.brillouin_signaller.measurement_result_ready.connect(self.handle_measurement_results)
        self.brillouin_signaller.zaber_lens_position_updated.connect(self.update_zaber_lens_position)
        self.brillouin_signaller.microwave_frequency_updated.connect(self.update_ref_freq_input)
        self.brillouin_signaller.calibration_result_ready.connect(self.handle_requested_calibration)
        self.brillouin_signaller.do_live_fitting_state.connect(self.update_do_live_fitting_checkbox)
        self.brillouin_signaller.gui_ready_received.connect(self.brillouin_signaller.on_gui_ready)
        self.brillouin_signaller.update_system_state_in_frontend.connect(self.update_system_state_label)
        self.brillouin_signaller.send_update_stored_axial_scans.connect(self.receive_axial_scan_list)
        self.brillouin_signaller.axial_scan_data_ready.connect(self.handle_received_axial_scan_data)

        # Saving Signals
        self.save_all_axial_scans_requested.connect(self.brillouin_signaller.save_all_axial_scans)
        self.save_selected_axial_scans_requested.connect(self.brillouin_signaller.save_multiple_axial_scans)
        self.remove_selected_axial_scans_requested.connect(self.brillouin_signaller.remove_selected_axial_scans)

        # Connect signals BEFORE starting the thread
        self.brillouin_signaller.log_message.connect(lambda msg: print("[Signaller]", msg))
        self.brillouin_signaller_thread.started.connect(self.run_gui)

        # Start the thread after all connections
        self.brillouin_signaller_thread.start()

        self.init_ui()

        self.update_gui()




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

        # RIGHT COLUMN: Plots
        middle_column_layout = QVBoxLayout()
        middle_column_layout.addWidget(self.create_andor_display_group())

        shift_controls_row = QHBoxLayout()
        shift_controls_row.addWidget(self.create_zaber_manual_movement_group())
        shift_controls_row.addWidget(self.create_take_axial_scan_group())

        middle_column_layout.addLayout(shift_controls_row)


        outer_layout.addLayout(middle_column_layout, 1)






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

        self.do_live_fitting_checkbox = QCheckBox("Do Live Fitting")
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
        group = QGroupBox("Allied Vision Cameras")
        layout = QVBoxLayout()

        self.allied_settings_btn = QPushButton("Settings")
        self.allied_settings_btn.clicked.connect(self.on_allied_settings_clicked)

        layout.addWidget(self.allied_settings_btn)
        group.setLayout(layout)
        return group



    def create_take_axial_scan_group(self):
        group = QGroupBox("Take Axial Scan")
        layout = QFormLayout()

        self.axial_id_input = QLineEdit()
        self.axial_num_input = QLineEdit("5")
        self.axial_step_input = QLineEdit("9")

        layout.addRow("ID:", self.axial_id_input)
        layout.addRow("Num Meas:", self.axial_num_input)
        layout.addRow("Step Size (µm):", self.axial_step_input)

        self.axial_btn = QPushButton("Take Axial Scan")
        self.axial_btn.clicked.connect(self.take_axial_scan)
        layout.addRow(self.axial_btn)

        group.setLayout(layout)
        return group

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

    def create_andor_display_group(self):
        group = QGroupBox("Andor Frame and Fitting")

        self.fig, (self.ax_img, self.ax_fit, self.ax_history) = plt.subplots(3, 1, figsize=(8, 8))
        self.fig.subplots_adjust(hspace=0.9)
        self.canvas = FigureCanvas(self.fig)

        import numpy as np
        dummy_frame = np.zeros((10, 10))
        self.img_artist = self.ax_img.imshow(
            dummy_frame,
            cmap="gray",
            aspect="equal",
            interpolation="none",
            origin="upper"
        )
        self._last_img_shape = dummy_frame.shape
        self.ax_img.set_xlabel("Pixel (X)")
        self.ax_img.set_ylabel("Pixel (Y)")

        # spectrum lines
        (self.spectrum_line,) = self.ax_fit.plot([], [], "k.", label="Spectrum")
        (self.fit_line,) = self.ax_fit.plot([], [], "r--", label="Fit")
        self.ax_fit.set_xlabel("Pixel (X)")
        self.ax_fit.set_ylabel("Intensity")

        # history line
        from collections import deque
        self.history_data = deque(maxlen=100)
        (self.history_line,) = self.ax_history.plot([], [], "b-")
        self.ax_history.set_title("Last 100 Fits (Frequency Shift)")
        self.ax_history.set_ylabel("GHz")
        self.ax_history.set_xlabel("Frame Index")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        group.setLayout(layout)
        group.setMinimumWidth(700)
        return group

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

        group.setLayout(layout)
        return group

    # ---------------- GUI Update Loop ---------------- #
    def on_stop_clicked(self):
        print("[Brillouin Viewer] STOP clicked.")
        self.brillouin_signaller.cancel_operations()
        self.close_all_shutters_requested.emit()
        self.stop_live_requested.emit()

    def on_cancel_event_clicked(self):
        print("[BrillouinViewer] Cancel button clicked.")
        # self.cancel_requested.emit()
        self.brillouin_signaller.cancel_operations()

    def on_restart_clicked(self):
        print("[Brillouin Viewer] Restart clicked.")
        self.stop_live_requested.emit()
        QApplication.processEvents()
        self.start_live_requested.emit()

    def update_gui(self):
        # Update the gui
        self.brillouin_signaller.update_gui()

    def save_all_axial_scans(self):
        self.save_all_axial_scans_requested.emit()

    def save_selected_axial_scan(self):
        selected_items = self.axial_scans_list.selectedItems()
        if not selected_items:
            print("[Warning] No scans selected.")
            return

        indices = [int(item.text().split(" - ")[0]) for item in selected_items]
        self.save_selected_axial_scans_requested.emit(indices)

    def move_zaber_lens_by(self, direction: int):
        try:
            step = float(self.lens_step_input.text())
            self.move_zaber_eye_lens_requested.emit(direction * step)
        except ValueError:
            print("[Error] Invalid lens step size input.")

    def remove_selected_axial_scan(self):
        selected_items = self.axial_scans_list.selectedItems()
        if not selected_items:
            print("[Warning] No scan selected.")
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

    def update_fitting_configs(self, fitting_configs: FittingConfigs):
        self.update_fitting_configs_requested.emit(fitting_configs)

    def on_fitting_configs_clicked(self):
        dialog = FindPeaksConfigDialog(on_apply=self.update_fitting_configs, parent=self)
        dialog.exec_()

    def on_reference_configs_clicked(self):
        dialog = CalibrationConfigDialog(self)
        dialog.exec_()

    def on_allied_settings_clicked(self):
        print("[Brillouin Viewer] Allied Vision Settings button clicked.")
        # TODO: Launch settings dialog when available

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

    # ---------------- GUI Update Loop ---------------- #

    def update_background_ui(self):
        self.brillouin_signaller.emit_do_background_subtraction()
        self.brillouin_signaller.emit_is_background_available()

    def run_gui(self):
        self.start_live_requested.emit()

    def run_one_gui_update(self):
        self.snap_requested.emit()

    def display_result(self, display_results: DisplayResults):
        frame = display_results.frame
        x_px = display_results.x_pixels
        spectrum = display_results.sline

        # --- Image Plot (efficient update if size unchanged) ---
        h, w = frame.shape
        if (h, w) != self._last_img_shape:
            self.ax_img.clear()
            self.img_artist = self.ax_img.imshow(
                frame,
                cmap="gray",
                aspect="equal",
                interpolation="none",
                origin="upper"
            )
            self.ax_img.set_xticks(np.arange(0, w, 10))
            self.ax_img.set_yticks(np.arange(0, h, 5))
            self.ax_img.set_xlabel("Pixel (X)")
            self.ax_img.set_ylabel("Pixel (Y)")
            self._last_img_shape = (h, w)
        else:
            self.img_artist.set_data(frame)

        # --- Spectrum Plot ---
        self.spectrum_line.set_data(x_px, spectrum)
        interpeak, freq_shift_ghz = None, None
        if display_results.is_fitting_available:
            self.fit_line.set_data(display_results.x_fit_refined, display_results.y_fit_refined)
            interpeak = display_results.inter_peak_distance
            freq_shift_ghz = display_results.freq_shift_ghz
            self.ax_fit.legend()
        else:
            self.fit_line.set_data([], [])
        self.ax_fit.relim()
        self.ax_fit.autoscale_view()

        # --- History Plot ---
        if display_results.freq_shift_ghz is not None:
            self.history_data.append(display_results.freq_shift_ghz)
            self.history_line.set_data(range(len(self.history_data)), list(self.history_data))
            self.ax_history.relim()
            self.ax_history.autoscale_view()

        # --- Spectrum Title ---
        if interpeak is not None and freq_shift_ghz is not None:
            title = f"Spectrum Fit | Interpeak: {interpeak:.2f} px / {freq_shift_ghz:.3f} GHz"
        elif interpeak is not None:
            title = f"Spectrum Fit | Interpeak: {interpeak:.2f} px / - GHz"
        elif freq_shift_ghz is not None:
            title = f"Spectrum Fit | Interpeak: - px / {freq_shift_ghz:.3f} GHz"
        else:
            title = "Spectrum Fit | Interpeak: - px / - GHz"
        self.ax_fit.set_title(title)

        # --- Redraw once ---
        self.canvas.draw_idle()
        self.brillouin_signaller.gui_ready_received.emit()

    # ---------------- Handlers ---------------- #

    def update_camera_shutter_button(self, is_open: bool):
        text = "Close" if is_open else "Open"
        self.toggle_camera_shutter_btn.setText(text)

    def populate_camera_ui(self, settings: dict):
        self.exposure_input.setText(str(settings["exposure"]))
        self.gain_input.setText(str(settings["gain"]))


    def apply_camera_settings(self):
        try:
            exposure = round(float(self.exposure_input.text()), ndigits=4)
            gain = int(self.gain_input.text())

            settings = {
                "exposure": exposure,
                "gain": gain,
            }

            self.apply_camera_settings_requested.emit(settings)  # ✅ thread-safe
            print("[Brillouin Viewer] Sent new camera settings to worker.")

            self.emit_camera_settings_requested.emit()  # ✅ thread-safe

        except Exception as e:
            print(f"[Brillouin Viewer] Failed to apply camera settings: {e}")


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

    def update_main_display(self, pixmap: QPixmap):
        self.allied_camera_display.setPixmap(pixmap)

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
                print(f"[✓] Background image saved to: {pkl_path}")

                # Save as HDF5
                h5_path = path if path.endswith(".h5") else path + ".h5"
                native_dict = dataclass_to_hdf5_native_dict(data)
                save_dict_to_hdf5(h5_path, native_dict)
                print(f"[✓] Background image saved as HDF5 to: {h5_path}")

            except Exception as e:
                print(f"[Brillouin Viewer] [Error] Failed to save background data: {e}")

            finally:
                self.brillouin_signaller.background_data_ready.disconnect(receive_data)

        self.brillouin_signaller.background_data_ready.connect(receive_data)
        self.brillouin_signaller.emit_background_data()



    def set_reference_freq(self):
        try:
            freq = float(self.ref_freq_input.text())
            self.update_microwave_freq_requested.emit(freq)
        except ValueError:
            print("[Brillouin Viewer] [Reference] Invalid frequency input.")

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
            print("[Brillouin Viewer] No axial available.")
            return

        i = int(selected_scan.split(" - ")[0])

        self.request_axial_scan_data.emit(i)

    def handle_received_axial_scan_data(self, scan_data: AxialScan):
        self.axial_viewer = AxialScanViewer(scan_data)
        self.axial_viewer.resize(900, 900)
        self.axial_viewer.show()



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
        print(f"[Brillouin Viewer] Calibration available")

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
                print("[Brillouin Viewer] Calibration plot displayed.")
            except Exception as e:
                print(f"[Brillouin Viewer] Failed to plot calibration: {e}")

        elif self._save_cali:
            if cali_data is None:
                print("[Brillouin Viewer] Failed to save data, no data available")
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
                print(f"[✓] Calibration data saved to {pkl_path}")

                # Save HDF5
                h5_path = base_path if base_path.endswith(".h5") else base_path + ".h5"
                hdf5_dict = dataclass_to_hdf5_native_dict(cali_data)
                save_dict_to_hdf5(h5_path, hdf5_dict)
                print(f"[✓] Calibration data saved as HDF5 to {h5_path}")

            except Exception as e:
                print(f"[Error] Failed to save calibration data: {e}")

        self._show_cali = False
        self._save_cali = True

    def take_axial_scan(self):
        try:
            id_str = self.axial_id_input.text().strip()
            n_meas = int(self.axial_num_input.text())
            step = float(self.axial_step_input.text())

            # Log info
            print(
                f"[Brillouin Viewer] Axial Scan Request | ID: {id_str}, N: {n_meas}, Step: {step} µm")

            # Stop live view before scanning
            self.stop_live_requested.emit()
            QApplication.processEvents()

            request = RequestAxialScan(
                id=id_str,
                n_measurements=n_meas,
                step_size_um=step,
            )

            self.take_axial_scan_requested.emit(request)

        except Exception as e:
            print(f"[Brillouin Viewer] Failed to initiate axial scan: {e}")


        except Exception as e:
            print(f"[Brillouin Viewer] Measurement setup failed: {e}")

    def handle_measurement_results(self, measurement_result):
        pass

    def save_measurements_to_file(self):
        if not self._stored_measurements:
            print("[Brillouin Viewer] No measurements to save.")
            return

        # Ask user for base file path
        base_path, _ = QFileDialog.getSaveFileName(
            self, "Save Measurements (base name, without extension)",
            filter="All Files (*)"
        )
        if not base_path:
            return

        try:
            # Save as Pickle
            pkl_path = base_path if base_path.endswith(".pkl") else base_path + ".pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(self._stored_measurements, f)
            print(f"[✓] Pickle saved to: {pkl_path}")

            # Save as HDF5
            h5_path = base_path if base_path.endswith(".h5") else base_path + ".h5"
            native_dict = dataclass_to_hdf5_native_dict(self._stored_measurements)
            save_dict_to_hdf5(h5_path, native_dict)
            print(f"[✓] HDF5 saved to: {h5_path}")

        except Exception as e:
            print(f"[Brillouin Viewer] [Error] Failed to save: {e}")

    def clear_measurements(self):
        self._stored_measurements.clear()
        self.measurement_series_label.setText("Stored Series: 0")
        print("[Brillouin Viewer] Cleared all stored measurement series.")

    def closeEvent(self, event):
        print("[Brillouin Viewer] Shutdown initiated...")

        # Stop live view
        self.brillouin_signaller._running = False
        self.brillouin_signaller._thread_active = False

        # Close worker directly
        self.brillouin_signaller.close()  # it's fine to block now — we're exiting

        # Kill the thread
        self.brillouin_signaller_thread.quit()
        self.brillouin_signaller_thread.wait(1000)  # wait max 1 second

        print("[Brillouin Viewer] GUI shutdown complete.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply global style sheet to reduce font size
    app.setStyleSheet("""
        * {
            font-size: 8pt;
        }
    """)
    viewer = BrillouinEyeViewerFrontend()
    viewer.show()
    sys.exit(app.exec_())
