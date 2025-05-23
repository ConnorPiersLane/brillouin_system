
import sys
import pickle
import numpy as np

from PyQt5.QtGui import QDoubleValidator, QIntValidator, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGroupBox, QLabel, QLineEdit,
    QFileDialog, QPushButton, QHBoxLayout, QFormLayout, QDialog, QVBoxLayout, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from brillouin_system.gui.manager.brillouin_manager import BrillouinManager
from brillouin_system.gui.signaller.brillouin_signaller import BrillouinSignaller
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
# from brillouin_system.devices.cameras.mako.allied_vision_camera import AlliedVisionCamera
from brillouin_system.devices.cameras.allied.dummy_mako_camera import DummyMakoCamera
from brillouin_system.devices.microwave_device import MicrowaveDummy
from brillouin_system.devices.shutter_device import ShutterManagerDummy
from brillouin_system.my_dataclasses.background_data import BackgroundData
from brillouin_system.my_dataclasses.calibration_data import CalibrationData
from brillouin_system.my_dataclasses.fitting_results import FittingResults
from brillouin_system.devices.zaber_linear_dummy import ZaberLinearDummy
from brillouin_system.my_dataclasses.measurement_data import MeasurementData
from brillouin_system.utils.calibrate_spectrometer import _linear_model

###
# Add other guis
from config_dialog import ConfigDialog



## Testing
brillouin_manager = BrillouinManager(
        camera=DummyCamera(),
    shutter_manager=ShutterManagerDummy('human_interface'),
    microwave=MicrowaveDummy(),
    zaber=ZaberLinearDummy(),
    mako_camera=DummyMakoCamera(),
    is_sample_illumination_continuous=True
)

#
# # Real
# brillouin_manager = BrillouinManager(
#         camera=IxonUltra(
#             index = 0,
#             temperature = -80, #"off"
#             fan_mode = "full",
#             x_start = 40, x_end  = 120,
#             y_start= 300, y_end  = 320,
#             vbin= 1, hbin  = 1,
#             exposure_time = 0.5,
#             gain  = 1,
#             verbose = True,
#         ),
#     shutter_manager=ShutterManager('human_interface'),
#     microwave=Microwave(),
#     zaber=ZaberLinearController(),
#     mako_camera=DummyMakoCamera(),
#     is_sample_illumination_continuous=True
# )




class BrillouinViewer(QWidget):

    # Signals
    gui_ready = pyqtSignal()
    apply_camera_settings_requested = pyqtSignal(dict)
    toggle_camera_shutter_requested = pyqtSignal()
    emit_camera_settings_requested = pyqtSignal()
    update_sd_requested = pyqtSignal(float)
    update_fsr_requested = pyqtSignal(float)
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
    run_calibration_requested = pyqtSignal(list)
    take_measurements_requested = pyqtSignal(int, str, float)
    toggle_save_images_requested = pyqtSignal(bool)
    shutdown_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Keep track of state:
        self._current_calibration: CalibrationData | None = None
        self._stored_measurements: list[MeasurementData] = []  # list of measurement series

        self.setWindowTitle("Brillouin Viewer (Live)")

        self.brillouin_signaller = BrillouinSignaller(manager=brillouin_manager)
        self.brillouin_signaller_thread = QThread()
        self.brillouin_signaller.moveToThread(self.brillouin_signaller_thread)

        # Sending signals
        self.update_fsr_requested.connect(self.brillouin_signaller.apply_free_spectral_range)
        self.update_sd_requested.connect(self.brillouin_signaller.apply_spectral_dispersion)
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
        self.take_measurements_requested.connect(self.brillouin_signaller.take_measurements)
        self.gui_ready.connect(self.brillouin_signaller.on_gui_ready)
        self.toggle_save_images_requested.connect(self.brillouin_signaller.set_save_images_state)
        self.shutdown_requested.connect(self.brillouin_signaller.close)

        # Receiving signals
        self.brillouin_signaller.background_subtraction_state.connect(self.update_bg_subtraction)
        self.brillouin_signaller.background_available_state.connect(self.handle_is_bg_available)
        self.brillouin_signaller.illumination_mode_state.connect(self.update_illumination_ui)
        self.brillouin_signaller.reference_mode_state.connect(self.update_reference_ui)
        self.brillouin_signaller.camera_settings_ready.connect(self.populate_camera_ui)
        self.brillouin_signaller.camera_shutter_state_changed.connect(self.update_camera_shutter_button)
        self.brillouin_signaller.frame_and_fit_ready.connect(self.display_result)
        self.brillouin_signaller.calibration_result_ready.connect(self.handle_calibration_results)
        self.brillouin_signaller.measurement_result_ready.connect(self.handle_measurement_results)
        self.brillouin_signaller.zaber_position_updated.connect(self.update_zaber_position)
        self.brillouin_signaller.microwave_frequency_updated.connect(self.update_ref_freq_input)

        self.brillouin_signaller.sd_updated.connect(self.update_sd_ui)
        self.brillouin_signaller.fsr_updated.connect(self.update_fsr_ui)

        # Mako Camera Stream
        self.brillouin_signaller.mako_frame_ready.connect(self.update_mako_camera_image)

        # 🔁 Connect signals BEFORE starting the thread
        self.brillouin_signaller.log_message.connect(lambda msg: print("[Worker]", msg))
        self.brillouin_signaller_thread.started.connect(self.run_gui)

        # ✅ Start the thread after all connections
        self.brillouin_signaller_thread.start()

        self.init_ui()

        self.update_gui()

        self.brillouin_signaller.start_mako_stream()

    def init_ui(self):
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        # --- Plot area + Allied Vision camera view (side-by-side) ---
        plot_row_layout = QHBoxLayout()

        self.fig, (self.ax_img, self.ax_fit) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.subplots_adjust(hspace=0.9)
        self.canvas = FigureCanvas(self.fig)
        plot_row_layout.addWidget(self.canvas)

        # Placeholder for Allied Vision camera
        self.allied_camera_display = QLabel("Allied Vision Camera")
        self.allied_camera_display.setFixedSize(int(640*0.9), int(480*0.9))
        self.allied_camera_display.setStyleSheet("background-color: black; color: white;")
        self.allied_camera_display.setAlignment(Qt.AlignCenter)
        plot_row_layout.addWidget(self.allied_camera_display)

        outer_layout.addLayout(plot_row_layout)

        # --- Bottom control row: all widgets in one horizontal line ---
        control_row_layout = QHBoxLayout()
        control_row_layout.addWidget(self.create_background_group())
        control_row_layout.addWidget(self.create_camera_group())
        control_row_layout.addWidget(self.create_illumination_group())
        control_row_layout.addWidget(self.create_reference_group())
        control_row_layout.addWidget(self.create_spectrometer_group())
        control_row_layout.addWidget(self.create_zaber_group())
        control_row_layout.addWidget(self.create_measurement_group())

        outer_layout.addLayout(control_row_layout)

    def update_gui(self):
        # Update the gui
        self.brillouin_signaller.emit_is_illumination_continuous()
        self.brillouin_signaller.emit_is_background_available()
        self.brillouin_signaller.emit_camera_settings()
        self.brillouin_signaller.emit_do_background_subtraction()
        self.brillouin_signaller.emit_spectral_dispersion()
        self.brillouin_signaller.emit_free_spectral_range()



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
        self.exposure_input.setFixedWidth(60)

        self.gain_input = QLineEdit()
        self.gain_input.setValidator(QIntValidator(0, 1000))
        self.gain_input.setFixedWidth(60)

        self.roi_input = QLineEdit()
        self.roi_input.setFixedWidth(60)

        self.apply_camera_btn = QPushButton("Apply")
        self.apply_camera_btn.setFixedWidth(60)
        self.apply_camera_btn.clicked.connect(self.apply_camera_settings)

        self.toggle_camera_shutter_btn = QPushButton("Close")
        self.toggle_camera_shutter_btn.setFixedWidth(60)
        self.toggle_camera_shutter_btn.clicked.connect(self.toggle_camera_shutter_requested.emit)

        layout = QFormLayout()
        layout.addRow("Exp. Time (s):", self.exposure_input)
        layout.addRow("Gain:", self.gain_input)
        layout.addRow("ROI:", self.roi_input)
        layout.addRow(self.toggle_camera_shutter_btn, self.apply_camera_btn)

        # btn_row = QHBoxLayout()
        # btn_row.addWidget(self.toggle_camera_shutter_btn)
        # btn_row.addWidget(self.apply_camera_btn)
        # layout.addRow("", btn_row)

        self.config_settings_btn = QPushButton("Configs")
        self.config_settings_btn.clicked.connect(self.on_configs_clicked)
        layout.addRow(self.config_settings_btn)

        group = QGroupBox("Andor and Configs")
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
        self.ref_freq_input.setFixedWidth(60)
        self.ref_freq_input.setText("5.5")  # Default value

        # NEW: Set button
        self.set_ref_btn = QPushButton("Set")
        self.set_ref_btn.clicked.connect(self.set_reference_freq)

        # Input and button layout
        form_layout = QFormLayout()
        form_layout.addRow("Ref. Freq. (GHz):", self.ref_freq_input)
        form_layout.addRow("", self.set_ref_btn)

        layout = QVBoxLayout()
        layout.addLayout(mode_row)
        layout.addLayout(form_layout)

        group = QGroupBox("Reference")
        group.setLayout(layout)
        return group

    def create_spectrometer_group(self):
        self.sd_input = QLineEdit()
        self.sd_input.setValidator(QDoubleValidator(0.0001, 100.0, 6))
        self.sd_input.setFixedWidth(80)
        self.sd_input.setText("0.0")
        self.sd_input.editingFinished.connect(self.on_sd_edited)

        self.fsr_input = QLineEdit()
        self.fsr_input.setValidator(QDoubleValidator(0.1, 100.0, 6))
        self.fsr_input.setFixedWidth(80)
        self.fsr_input.setText("0.0")
        self.fsr_input.editingFinished.connect(self.on_fsr_edited)

        self.calib_freq_input = QLineEdit()
        self.calib_freq_input.setFixedWidth(80)
        self.calib_freq_input.setText("3.0, 4.0, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 7.0")

        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self.run_calibration)

        self.show_calib_btn = QPushButton("Show")
        self.show_calib_btn.clicked.connect(self.show_calibration_data)

        self.save_calib_btn = QPushButton("Save")
        self.save_calib_btn.clicked.connect(self.save_calibration_data)

        layout = QFormLayout()
        layout.addRow("SD (GHz/px):", self.sd_input)
        layout.addRow("FSR (GHz):", self.fsr_input)
        layout.addRow("Calibration Freqs [GHz]:", self.calib_freq_input)
        layout.addRow("", self.calibrate_btn)
        layout.addRow(self.show_calib_btn, self.save_calib_btn)
        group = QGroupBox("Spectrometer")
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

        self.num_images_input = QLineEdit("10")
        self.num_images_input.setValidator(QIntValidator(1, 9999))
        form_layout.addRow("Number of Measurements:", self.num_images_input)


        layout.addLayout(form_layout)

        self.measure_btn = QPushButton("Take Measurements")
        self.measure_btn.clicked.connect(self.take_measurements)
        layout.addWidget(self.measure_btn)

        self.measurement_series_label = QLabel("Stored Series: 0")
        # layout.addWidget(self.measurement_series_label)
        self.save_images_checkbox = QCheckBox("Save Images")
        self.save_images_checkbox.setChecked(True)
        self.save_images_checkbox.stateChanged.connect(self.on_save_images_toggled)

        row = QHBoxLayout()
        row.addWidget(self.measurement_series_label)
        # row.addStretch()
        row.addWidget(self.save_images_checkbox)

        layout.addLayout(row)

        self.save_measurement_series_btn = QPushButton("Save Measurements")
        self.save_measurement_series_btn.clicked.connect(self.save_measurements_to_file)
        layout.addWidget(self.save_measurement_series_btn)

        self.clear_measurement_series_btn = QPushButton("Clear Measurements")
        self.clear_measurement_series_btn.clicked.connect(self.clear_measurements)
        layout.addWidget(self.clear_measurement_series_btn)

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


    def update_sd_ui(self, sd: float):
        self.sd_input.setText(f"{sd:.3f}")

    def update_fsr_ui(self, fsr: float):
        self.fsr_input.setText(f"{fsr:.3f}")

    def on_sd_edited(self):
        try:
            sd = float(self.sd_input.text())
            self.update_sd_requested.emit(sd)
        except ValueError:
            print("[Brillouin Viewer] Invalid SD input.")

    def on_fsr_edited(self):
        try:
            fsr = float(self.fsr_input.text())
            self.update_fsr_requested.emit(fsr)  # ✅ thread-safe
        except ValueError:
            print("[Brillouin Viewer] Invalid FSR input.")

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

    def display_result(self, fitting: FittingResults):
        frame = fitting.frame
        spectrum = fitting.fitted_spectrum.sline
        interpeak = fitting.fitted_spectrum.inter_peak_distance
        freq_shift_ghz = fitting.freq_shift_ghz

        x_fit_refined = fitting.fitted_spectrum.x_fit_refined
        y_fit_refined = fitting.fitted_spectrum.y_fit_refined

        row_idx = np.argmax(frame.sum(axis=1))
        x_px = np.arange(len(spectrum))

        # --- Image Plot ---
        self.ax_img.clear()
        self.ax_img.imshow(frame, cmap="gray", aspect='equal', interpolation='none', origin="upper")
        self.ax_img.set_title(f"Camera Frame | Selected Middle Row: {row_idx}")
        self.ax_img.set_xticks(np.arange(0, frame.shape[1], 10))
        self.ax_img.set_yticks(np.arange(0, frame.shape[0], 5))
        self.ax_img.set_xlabel("Pixel (X)")
        self.ax_img.set_ylabel("Pixel (Y)")

        # --- Spectrum Plot ---
        self.ax_fit.clear()
        self.ax_fit.plot(x_px, spectrum, 'k.', label="Spectrum")

        if interpeak is not np.nan:
            self.ax_fit.plot(x_fit_refined, y_fit_refined, 'r--', label="Fit")
            if freq_shift_ghz is not None:
                self.ax_fit.set_title(f"Spectrum Fit | Interpeak: {interpeak:.2f} px / {freq_shift_ghz:.3f} GHz")
            else:
                self.ax_fit.set_title(f"Spectrum Fit | Interpeak: {interpeak:.2f} px / - GHz")
            self.ax_fit.legend()

        else:
            self.ax_fit.set_title("Spectrum Fit | Interpeak: - px / - GHz")

        self.canvas.draw()
        self.gui_ready.emit()


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
        x0, x1, y0, y1 = settings["roi"]
        hbin, vbin = settings["binning"]
        roi_str = f"{x0},{x1},{y0},{y1},{hbin},{vbin}"
        self.roi_input.setText(roi_str)


    def apply_camera_settings(self):
        try:
            exposure = round(float(self.exposure_input.text()), ndigits=4)
            gain = int(self.gain_input.text())

            parts = [int(p.strip()) for p in self.roi_input.text().split(",")]
            if len(parts) != 6:
                raise ValueError("ROI must have 6 comma-separated integers.")

            roi = parts[:4]
            binning = parts[4:]

            settings = {
                "exposure": exposure,
                "gain": gain,
                "roi": roi,
                "binning": binning,
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
            self.calibrate_btn.setEnabled(True)
            self.save_calib_btn.setEnabled(True)
        else:
            self.calib_label_meas.setText("● Meas.")
            self.calib_label_meas.setStyleSheet("color: green; font-weight: bold")
            self.calib_label_calib.setText("○ Ref.")
            self.calib_label_calib.setStyleSheet("color: gray")
            self.calibrate_btn.setEnabled(False)
            self.save_calib_btn.setEnabled(False)


    def update_spectrometer_params(self):
        try:
            sd = float(self.sd_input.text())
            fsr = float(self.fsr_input.text())
            # Do something with sd and fsr
            print(f"[Brillouin Viewer] Updated values: SD = {sd}, FSR = {fsr}")
        except ValueError:
            print("[Brillouin Viewer] Invalid SD or FSR input.")

    def update_ref_freq_input(self, freq: float):
        self.ref_freq_input.setText(f"{freq:.3f}")

    # -------------- Functions --------------

    def update_mako_camera_image(self, frame: np.ndarray):
        try:
            if frame.ndim == 3 and frame.shape[-1] == 1:
                frame = frame[..., 0]
            norm = (frame - frame.min()) / (np.ptp(frame) + 1e-6)
            norm = (norm * 255).astype(np.uint8)
            h, w = norm.shape
            qimg = QImage(norm.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.allied_camera_display.width(),
                self.allied_camera_display.height(),
                Qt.KeepAspectRatio,
            )
            self.allied_camera_display.setPixmap(pixmap)
        except Exception as e:
            print(f"[AVCamera] Display failed: {e}")

    def save_background_image(self):
        def receive_data(data: BackgroundData):
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Background Image", filter="Pickle Files (*.pkl);;All Files (*)"
            )
            if not path:
                return

            try:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
                print(f"[Brillouin Viewer] Background image and settings saved to {path}")
            except Exception as e:
                print(f"[Brillouin Viewer] [Error] Failed to save background data: {e}")
            finally:
                self.brillouin_signaller.background_data_ready.disconnect(receive_data)

        self.brillouin_signaller.background_data_ready.connect(receive_data)
        self.brillouin_signaller.emit_background_data()

    def move_zaber(self, direction: int):
        try:
            step = float(self.zaber_step_input.text())
            axis = self.zaber_axis_selector.currentData()  # 'x', 'y', or 'z'
            delta = direction * step

            self.zaber_controller.move_rel(axis, delta)
            pos = self.zaber_controller.get_position(axis)
            self.pos_display.setText(f"{pos:.0f} µm")

        except Exception as e:
            print(f"[Zaber] Movement failed: {e}")

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
        self.stop_live_requested.emit()

        # Show wait dialog
        wait_dialog = QDialog(self)
        wait_dialog.setWindowTitle("Acquiring Background")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Please wait while the background image is being acquired..."))
        wait_dialog.setLayout(layout)
        wait_dialog.setModal(True)
        wait_dialog.setWindowFlags(wait_dialog.windowFlags() | Qt.WindowStaysOnTopHint)

        def on_bg_ready(data: BackgroundData):
            self.brillouin_signaller.background_data_ready.disconnect(on_bg_ready)
            wait_dialog.close()
            print("[Brillouin Viewer] Background image acquired.")
            self.update_background_ui()
            self.start_live_requested.emit()

        self.brillouin_signaller.background_data_ready.connect(on_bg_ready)

        wait_dialog.show()
        self.acquire_background_requested.emit()


    def open_file_dialog(self, event):
        path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", filter="Pickle Files (*.pkl);;All Files (*)"
        )
        if path:
            self.file_path_input.setText(path)


    def run_calibration(self):
        try:
            freqs = [float(f) for f in self.calib_freq_input.text().replace(",", " ").split()]
            if freqs:
                self.run_calibration_requested.emit(freqs)
            else:
                print("[Brillouin Viewer] No calibration frequencies provided.")
        except ValueError:
            print("[Brillouin Viewer] Invalid calibration input.")

    def handle_calibration_results(self, data: CalibrationData):
        self._current_calibration = data
        print(f"[Brillouin Viewer] Calibration data stored")

    def show_calibration_data(self):
        if self._current_calibration is None:
            print("[Brillouin Viewer] No calibration data to show.")
            return


        data: CalibrationData = self._current_calibration
        px_dists = [fit.inter_peak_distance_px for fit in data.data]
        freqs = data.reference_freqs_ghz

        # Fitted line
        x_fit = np.linspace(min(px_dists), max(px_dists), 100)
        y_fit = _linear_model(x=x_fit, sd=data.sd, fsr=data.fsr)

        # Create a new window with a matplotlib plot
        win = QDialog(self)
        win.setWindowTitle("Calibration Fit")
        layout = QVBoxLayout(win)

        fig, ax = plt.subplots()
        ax.plot(px_dists, freqs, 'ko', label="Data")
        ax.plot(x_fit, y_fit, 'r--', label=f"Fit: SD={data.sd:.4f}, FSR={data.fsr:.2f}")
        ax.set_xlabel("Pixel Distance")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title("Calibration Curve")
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        win.setLayout(layout)
        win.resize(600, 400)
        win.show()
    def save_calibration_data(self):

        data = self._current_calibration
        if data is None:
            print(f"[Brillouin Viewer] Failed to save data, no data availalbe")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration Data", filter="Pickle Files (*.pkl);;All Files (*)"
        )
        if not path:
            return

        if path:
            try:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
                print(f"[✓] Calibration data saved to {path}")
            except Exception as e:
                print(f"[Error] Failed to save calibration data: {e}")


    def take_measurements(self):
        try:
            n = int(self.num_images_input.text())
            auto_move = self.move_stage_checkbox.isChecked()
            step_size = float(self.zaber_step_input.text())
            if auto_move:
                axis = self.zaber_axis_selector.currentData()
            else:
                axis = ''

            self.stop_live_requested.emit()
            QApplication.processEvents()

            self.take_measurements_requested.emit(n, axis, step_size)

        except Exception as e:
            print(f"[Brillouin Viewer] Measurement setup failed: {e}")

    def handle_measurement_results(self, measurement_result: MeasurementData):
        self._stored_measurements.append(measurement_result)
        self.measurement_series_label.setText(f"Stored Series: {len(self._stored_measurements)}")
        self.start_live_requested.emit()

    def on_save_images_toggled(self, state: int):
        self.toggle_save_images_requested.emit(bool(state))

    def save_measurements_to_file(self):
        if not self._stored_measurements:
            print("[Brillouin Viewer] No measurements to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Measurements", filter="Pickle Files (*.pkl);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "wb") as f:
                pickle.dump(self._stored_measurements, f)
            print(f"[✓] Measurement series saved to {path}")
        except Exception as e:
            print(f"[Brillouin Viewer] [Error] Failed to save measurements: {e}")

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
