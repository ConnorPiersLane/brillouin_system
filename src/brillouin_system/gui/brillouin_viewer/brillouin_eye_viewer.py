
import sys
import pickle
import numpy as np

from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGroupBox, QLabel, QLineEdit,
    QFileDialog, QPushButton, QHBoxLayout, QFormLayout, QVBoxLayout, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from brillouin_system.config.config import calibration_config
from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
from brillouin_system.devices.zaber_linear import ZaberLinearController
from brillouin_system.gui.brillouin_viewer.brillouin_backend import BrillouinBackend
from brillouin_system.gui.brillouin_viewer.brillouin_signaller import BrillouinSignaller
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
# from brillouin_system.devices.cameras.mako.allied_vision_camera import AlliedVisionCamera
from brillouin_system.devices.microwave_device import MicrowaveDummy, Microwave
from brillouin_system.devices.shutter_device import ShutterManagerDummy, ShutterManager

from brillouin_system.my_dataclasses.background_image import BackgroundImage
from brillouin_system.my_dataclasses.measurements import MeasurementSettings
from brillouin_system.my_dataclasses.calibration import render_calibration_to_pixmap, \
    CalibrationImageDialog, CalibrationData, CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_results import DisplayResults
from brillouin_system.devices.zaber_linear import ZaberLinearDummy
from brillouin_system.my_dataclasses.measurements import MeasurementSeries


###
# Add other guis
from brillouin_system.config.config_dialog import ConfigDialog
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import dataclass_to_hdf5_native_dict, save_dict_to_hdf5

## Testing
brillouin_manager = BrillouinBackend(
        camera=DummyCamera(),
    shutter_manager=ShutterManagerDummy('human_interface'),
    microwave=MicrowaveDummy(),
    zaber=ZaberLinearDummy(),
    is_sample_illumination_continuous=True
)


# # Real
brillouin_manager = BrillouinBackend(
        camera=IxonUltra(
            index = 0,
            temperature = -80, #"off"
            fan_mode = "full",
            x_start = 40, x_end  = 120,
            y_start= 300, y_end  = 315,
            vbin= 1, hbin  = 1,
            verbose = True,
            advanced_gain_option=False
        ),
    shutter_manager=ShutterManager('human_interface'),
    microwave=Microwave(),
    zaber=ZaberLinearController(),
    is_sample_illumination_continuous=True
)


class BrillouinViewer(QWidget):

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
    move_zaber_requested = pyqtSignal(str, float)
    request_zaber_position = pyqtSignal(str)
    run_calibration_requested = pyqtSignal()
    take_measurement_requested = pyqtSignal(object)
    shutdown_requested = pyqtSignal()
    get_calibration_results_requested = pyqtSignal()
    toggle_do_live_fitting_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        self._stored_measurements: list[MeasurementSeries] = []  # list of measurement series

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
        self.move_zaber_requested.connect(self.brillouin_signaller.move_zaber_relative)
        self.request_zaber_position.connect(self.brillouin_signaller.query_zaber_position)
        self.run_calibration_requested.connect(self.brillouin_signaller.run_calibration)
        self.take_measurement_requested.connect(self.brillouin_signaller.take_measurements)
        self.gui_ready.connect(self.brillouin_signaller.on_gui_ready)
        self.shutdown_requested.connect(self.brillouin_signaller.close)
        self.get_calibration_results_requested.connect(self.brillouin_signaller.get_calibration_results)
        self.toggle_do_live_fitting_requested.connect(self.brillouin_signaller.toggle_do_live_fitting)
        self.cancel_requested.connect(self.brillouin_signaller.cancel_operations)

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
        self.brillouin_signaller.zaber_position_updated.connect(self.update_zaber_position)
        self.brillouin_signaller.microwave_frequency_updated.connect(self.update_ref_freq_input)
        self.brillouin_signaller.calibration_result_ready.connect(self.handle_requested_calibration)
        self.brillouin_signaller.do_live_fitting_state.connect(self.update_do_live_fitting_checkbox)
        self.brillouin_signaller.gui_ready_received.connect(self.brillouin_signaller.on_gui_ready)

        # Connect signals BEFORE starting the thread
        self.brillouin_signaller.log_message.connect(lambda msg: print("[Signaller]", msg))
        self.brillouin_signaller_thread.started.connect(self.run_gui)

        # Start the thread after all connections
        self.brillouin_signaller_thread.start()

        self.init_ui()

        self.update_gui()


    def init_ui(self):
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        # --- Plot area + Main Display area (side-by-side) ---
        plot_row_layout = QHBoxLayout()

        self.fig, (self.ax_img, self.ax_fit) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.subplots_adjust(hspace=0.9)
        self.canvas = FigureCanvas(self.fig)
        plot_row_layout.addWidget(self.canvas)

        # Placeholder for Allied Vision camera
        self.main_display = QLabel("Diplay area")
        self.main_display.setFixedSize(int(640 * 0.9), int(480 * 0.9))
        self.main_display.setStyleSheet("background-color: black; color: white;")
        self.main_display.setAlignment(Qt.AlignCenter)
        plot_row_layout.addWidget(self.main_display)

        outer_layout.addLayout(plot_row_layout)

        # --- Bottom control row: all widgets in one horizontal line ---
        control_row_layout = QHBoxLayout()
        control_row_layout.addWidget(self.create_background_group())
        control_row_layout.addWidget(self.create_camera_group())
        control_row_layout.addWidget(self.create_illumination_group())
        control_row_layout.addWidget(self.create_reference_group())
        control_row_layout.addWidget(self.create_config_group())
        control_row_layout.addWidget(self.create_zaber_group())
        control_row_layout.addWidget(self.create_measurement_group())

        outer_layout.addLayout(control_row_layout)

    def update_gui(self):
        # Update the gui
        self.brillouin_signaller.update_gui()




    # ---------------- UI Sections ---------------- #

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

        layout = QVBoxLayout()
        layout.addWidget(self.bg_label_off)
        layout.addWidget(self.bg_label_on)
        layout.addSpacing(5)
        layout.addWidget(self.btn_take_bg)
        layout.addWidget(self.toggle_bg_btn)
        layout.addWidget(self.btn_save_bg)

        group = QGroupBox("Background")
        group.setLayout(layout)
        return group

    def create_camera_group(self):
        self.exposure_input = QLineEdit()
        self.exposure_input.setValidator(QDoubleValidator(0.001, 60.0, 3))

        self.gain_input = QLineEdit()
        self.gain_input.setValidator(QIntValidator(0, 1000))

        self.apply_camera_btn = QPushButton("Apply")
        self.apply_camera_btn.clicked.connect(self.apply_camera_settings)

        self.toggle_camera_shutter_btn = QPushButton("Close")
        self.toggle_camera_shutter_btn.clicked.connect(self.toggle_camera_shutter_requested.emit)

        self.do_live_fitting_checkbox = QCheckBox("Do Live Fitting")
        self.do_live_fitting_checkbox.stateChanged.connect(self.on_do_live_fitting_toggled)

        layout = QFormLayout()
        layout.addRow("Exp. Time (s):", self.exposure_input)
        layout.addRow("Gain:", self.gain_input)
        layout.addRow(self.toggle_camera_shutter_btn, self.apply_camera_btn)
        layout.addRow(self.do_live_fitting_checkbox)

        # btn_row = QHBoxLayout()
        # btn_row.addWidget(self.toggle_camera_shutter_btn)
        # btn_row.addWidget(self.apply_camera_btn)
        # layout.addRow("", btn_row)

        group = QGroupBox("Andor Camera")
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


        layout = QVBoxLayout()
        layout.addWidget(self.illum_label_cont)
        layout.addWidget(self.illum_label_pulse)
        layout.addSpacing(5)
        layout.addWidget(self.toggle_illum_btn)
        layout.addWidget(self.snap_once_btn)

        group = QGroupBox("Illumination")
        group.setLayout(layout)
        return group


    def create_reference_group(self):
        # Mode labels stacked vertically
        self.calib_label_meas = QLabel("● Meas.")
        self.calib_label_calib = QLabel("○ Ref.")
        self.calib_label_meas.setStyleSheet("color: green; font-weight: bold")
        self.calib_label_calib.setStyleSheet("color: gray")

        label_col = QVBoxLayout()
        label_col.addWidget(self.calib_label_meas)
        label_col.addWidget(self.calib_label_calib)

        # Mode switch button
        self.toggle_calib_btn = QPushButton("Switch")
        self.toggle_calib_btn.clicked.connect(self.toggle_reference_mode)

        mode_row = QHBoxLayout()
        mode_row.addLayout(label_col)
        mode_row.addWidget(self.toggle_calib_btn)

        # Ref. Freq input
        self.ref_freq_input = QLineEdit()
        self.ref_freq_input.setValidator(QDoubleValidator(0.0, 100.0, 4))
        self.ref_freq_input.setText("5.5")  # Default value

        # NEW: Set button
        self.set_ref_btn = QPushButton("Set freq")
        self.set_ref_btn.clicked.connect(self.set_reference_freq)

        self.calibrate_btn = QPushButton("Cal. all")
        self.calibrate_btn.clicked.connect(self.run_calibration)

        self.show_calib_btn = QPushButton("Show")
        self.show_calib_btn.clicked.connect(self.show_calibration_results)
        self.show_calib_btn.setEnabled(False)

        self.save_calib_btn = QPushButton("Save")
        self.save_calib_btn.clicked.connect(self.save_calibration_results)
        self.save_calib_btn.setEnabled(False)

        # Input and button layout
        form_layout = QFormLayout()
        form_layout.addRow("Ref. Freq. (GHz):", self.ref_freq_input)
        form_layout.addRow(self.calibrate_btn, self.set_ref_btn)
        form_layout.addRow(self.show_calib_btn, self.save_calib_btn)

        layout = QVBoxLayout()
        layout.addLayout(mode_row)
        layout.addLayout(form_layout)

        group = QGroupBox("Reference")
        group.setLayout(layout)
        return group

    def create_config_group(self):
        group = QGroupBox("More Configs")
        layout = QVBoxLayout()

        self.config_settings_btn = QPushButton("Configs")
        self.config_settings_btn.clicked.connect(self.on_configs_clicked)
        layout.addWidget(self.config_settings_btn)

        group.setLayout(layout)
        return group

    def create_zaber_group(self):
        self.zaber_step_input = QLineEdit("100")
        self.zaber_step_input.setValidator(QDoubleValidator(0.1, 100000.0, 3))
        self.zaber_step_input.setFixedWidth(60)

        self.pos_display = QLabel("0.00 µm")
        self.pos_display.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.pos_display.setStyleSheet("font-weight: bold; padding-left: 8px;")

        # --- New Axis Dropdown ---
        self.zaber_axis_selector = QComboBox()
        self.zaber_axis_selector.setFixedWidth(60)
        self.zaber_axis_selector.currentIndexChanged.connect(self.on_zaber_axis_changed)
        # Request initial position display
        if self.zaber_axis_selector.count() > 0:
            default_axis = self.zaber_axis_selector.currentData()
            self.request_zaber_position.emit(default_axis)

        # Populate axis selector from controller
        try:
            available_axes = self.brillouin_signaller.manager.zaber.get_available_axes()  # ['x', 'y']
            for axis in available_axes:
                self.zaber_axis_selector.addItem(axis.upper(), axis)  # Display 'X', 'Y'; store 'x', 'y'
        except Exception as e:
            print(f"[Zaber] Failed to populate axis selector: {e}")

        # --- Arrow buttons ---
        self.left_btn = QPushButton("←")
        self.right_btn = QPushButton("→")
        self.left_btn.setFixedWidth(50)
        self.right_btn.setFixedWidth(50)
        self.left_btn.clicked.connect(lambda: self.move_zaber(-1))
        self.right_btn.clicked.connect(lambda: self.move_zaber(+1))

        # --- Movement direction ---
        self.move_direction_box = QComboBox()
        self.move_direction_box.addItems(["Forward", "Backward"])
        self.move_direction_box.setFixedWidth(100)

        self.move_stage_checkbox = QCheckBox("Move Zaber between measurements")

        # --- Layouts ---

        # Row 1: Axis selector + arrows
        row0 = QHBoxLayout()
        row0.addWidget(self.zaber_axis_selector)
        row0.addWidget(self.left_btn)
        row0.addWidget(self.right_btn)
        row0.addStretch()

        # Row 2: Step size
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Step Size (µm):"))
        row1.addWidget(self.zaber_step_input)
        row1.addStretch()

        # Row 3: Position
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Position (µm):"))
        row2.addWidget(self.pos_display)
        row2.addStretch()

        # Row 4: Direction
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Direction:"))
        row3.addWidget(self.move_direction_box)
        row3.addStretch()

        # Final layout
        layout = QVBoxLayout()
        layout.addLayout(row0)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addSpacing(5)
        layout.addLayout(row3)
        layout.addWidget(self.move_stage_checkbox)

        group = QGroupBox("Zaber Motion")
        group.setLayout(layout)
        return group

    def create_measurement_group(self):
        group = QGroupBox("Measurement Settings")
        layout = QVBoxLayout()

        form_layout = QFormLayout()

        # Number of Measurements
        self.num_images_input = QLineEdit("10")
        self.num_images_input.setValidator(QIntValidator(1, 9999))
        form_layout.addRow("Number of Measurements:", self.num_images_input)

        # Name of Series
        self.series_name_input = QLineEdit()
        form_layout.addRow("Name:", self.series_name_input)

        # Power input
        self.power_input = QLineEdit()
        self.power_input.setValidator(QDoubleValidator(0.0, 1000.0, 2))
        form_layout.addRow("Power [mW]:", self.power_input)

        layout.addLayout(form_layout)

        # --- Take + Cancel Buttons
        self.measure_btn = QPushButton("Take")
        self.measure_btn.clicked.connect(self.take_measurements)

        self.cancel_event_btn = QPushButton("Cancel")
        self.cancel_event_btn.clicked.connect(self.on_cancel_event_clicked)

        take_row = QHBoxLayout()
        take_row.addWidget(self.measure_btn)
        take_row.addWidget(self.cancel_event_btn)
        layout.addLayout(take_row)

        # Measurement info label
        self.measurement_series_label = QLabel("Stored Series: 0")
        layout.addWidget(self.measurement_series_label)

        # --- Save + Clear Buttons
        self.save_measurement_series_btn = QPushButton("Save")
        self.save_measurement_series_btn.clicked.connect(self.save_measurements_to_file)

        self.clear_measurement_series_btn = QPushButton("Clear")
        self.clear_measurement_series_btn.clicked.connect(self.clear_measurements)

        save_clear_row = QHBoxLayout()
        save_clear_row.addWidget(self.save_measurement_series_btn)
        save_clear_row.addWidget(self.clear_measurement_series_btn)
        layout.addLayout(save_clear_row)

        group.setLayout(layout)
        return group

    # ---------------- Signal Handles ---------------- #
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

    def on_cancel_event_clicked(self):
        print("[BrillouinViewer] Cancel button clicked.")
        self.cancel_requested.emit()

    # ---------------- Toggle ---------------- #
    def toggle_background_subtraction(self):
        self.toggle_bg_subtraction_requested.emit()

    def toggle_illumination(self):
        self.toggle_illumination_requested.emit()

    def on_configs_clicked(self):
        dialog = ConfigDialog(self)
        if dialog.exec_():
            settings = dialog.get_settings()
            print("[Brillouin Viewer] Received Configs:", settings)

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


        # --- Image Plot ---
        self.ax_img.clear()
        self.ax_img.imshow(frame, cmap="gray", aspect='equal', interpolation='none', origin="upper")
        self.ax_img.set_xticks(np.arange(0, frame.shape[1], 10))
        self.ax_img.set_yticks(np.arange(0, frame.shape[0], 5))
        self.ax_img.set_xlabel("Pixel (X)")
        self.ax_img.set_ylabel("Pixel (Y)")

        # --- Spectrum Plot ---
        self.ax_fit.clear()
        self.ax_fit.plot(x_px, spectrum, 'k.', label="Spectrum")

        interpeak = None
        freq_shift_ghz = None

        if display_results.is_fitting_available:
            x_fit_refined = display_results.x_fit_refined
            y_fit_refined = display_results.y_fit_refined
            interpeak = display_results.inter_peak_distance
            freq_shift_ghz = display_results.freq_shift_ghz
            self.ax_fit.plot(x_fit_refined, y_fit_refined, 'r--', label="Fit")
            self.ax_fit.legend()

        if interpeak is not None and freq_shift_ghz is not None:
            title = f"Spectrum Fit | Interpeak: {interpeak:.2f} px / {freq_shift_ghz:.3f} GHz"
        elif interpeak is not None:
            title = f"Spectrum Fit | Interpeak: {interpeak:.2f} px / - GHz"
        elif freq_shift_ghz is not None:
            title = f"Spectrum Fit | Interpeak: - px / {freq_shift_ghz:.3f} GHz"
        else:
            title = "Spectrum Fit | Interpeak: - px / - GHz"

        self.ax_fit.set_title(title)
        self.canvas.draw()
        self.brillouin_signaller.gui_ready_received.emit()


    def update_zaber_position(self, pos: float):
        self.pos_display.setText(f"{pos:.0f} µm")

    def on_zaber_axis_changed(self, index: int):
        axis = self.zaber_axis_selector.itemData(index)
        self.request_zaber_position.emit(axis)

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


    def move_zaber(self, direction):
        try:
            axis = self.zaber_axis_selector.currentData()
            step = float(self.zaber_step_input.text())
            delta = direction * step
            self.move_zaber_requested.emit(axis, delta)
        except Exception as e:
            print(f"[Brillouin Viewer] [Zaber] Movement request failed: {e}")

    def set_reference_freq(self):
        try:
            freq = float(self.ref_freq_input.text())
            self.update_microwave_freq_requested.emit(freq)
        except ValueError:
            print("[Brillouin Viewer] [Reference] Invalid frequency input.")

    def take_background_image(self):
        self.acquire_background_requested.emit()



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

    def take_measurements(self):
        try:
            n = int(self.num_images_input.text())
            axis = self.zaber_axis_selector.currentData()

            if self.move_stage_checkbox.isChecked():
                step_size = float(self.zaber_step_input.text())
            else:
                step_size = 0.0


            name = self.series_name_input.text().strip()
            power = float(self.power_input.text())

            print(f"[Brillouin Viewer] Preparing to take {n} measurements along {axis}-axis "
                  f"with step {step_size} µm. Name: '{name}', Power: {power:.2f} mW")

            self.stop_live_requested.emit()
            QApplication.processEvents()

            measurement_settings = MeasurementSettings(
                name = name,
                n_measurements=n,
                power_mW=power,
                move_axes = 'x',
                move_x_rel_um = step_size,
                move_y_rel_um = 0.0,
                move_z_rel_um = 0.0,
            )


            # If needed, you can pass name and power through a signal or store them for later use.
            self.take_measurement_requested.emit(measurement_settings)

        except Exception as e:
            print(f"[Brillouin Viewer] Measurement setup failed: {e}")

    def handle_measurement_results(self, measurement_result: MeasurementSeries):
        self._stored_measurements.append(measurement_result)
        self.measurement_series_label.setText(f"Stored Series: {len(self._stored_measurements)}")
        self.start_live_requested.emit()

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
    viewer = BrillouinViewer()
    viewer.show()
    sys.exit(app.exec_())
