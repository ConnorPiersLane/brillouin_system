
import sys
import pickle
import numpy as np

from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGroupBox, QLabel, QLineEdit,
    QFileDialog, QPushButton, QHBoxLayout, QFormLayout, QVBoxLayout, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from brillouin_system.devices.cameras.andor.andor_frame.andor_config_dialog import AndorConfigDialog
from brillouin_system.calibration.config.calibration_config_gui import CalibrationConfigDialog
from brillouin_system.config.config import calibration_config
from brillouin_system.spectrum_fitting.peak_fitting import FindPeaksConfigDialog
from brillouin_system.devices.cameras.flir.flir_config.flir_config import FLIRConfig
from brillouin_system.devices.cameras.flir.flir_dummy import DummyFLIRCamera
from brillouin_system.devices.cameras.flir.flir_worker import FlirWorker
from brillouin_system.devices.zaber_engines.zaber_microscope.led_config.led_config import LEDConfig
from brillouin_system.devices.zaber_engines.zaber_microscope.led_config.led_config_dialog import LEDConfigDialog
from brillouin_system.devices.zaber_engines.zaber_microscope.zaber_microscope import DummyZaberMicroscope
from brillouin_system.gui.human_interface.brillouin_backend import BrillouinBackend
from brillouin_system.gui.human_interface.brillouin_signaller import BrillouinSignaller, SystemState
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
# from brillouin_system.devices.cameras.mako.allied_vision_camera import AlliedVisionCamera
from brillouin_system.devices.microwave_device import MicrowaveDummy
from brillouin_system.devices.shutter_device import ShutterManagerDummy

from brillouin_system.my_dataclasses.background_image import BackgroundImage
from brillouin_system.calibration.calibration import render_calibration_to_pixmap, \
    CalibrationImageDialog, CalibrationData, CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_spectrum import DisplayResults

# SubGuis
from brillouin_system.devices.cameras.flir.flir_config.flir_config_dialog import FLIRConfigDialog


###
# Add other guis
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import dataclass_to_hdf5_native_dict, save_dict_to_hdf5




## Testing
brillouin_backend = BrillouinBackend(
    system_type='microscope',
        camera=DummyCamera(),
    shutter_manager=ShutterManagerDummy('human_interface'),
    microwave=MicrowaveDummy(),
    zaber_microscope=DummyZaberMicroscope(),
    flir_cam_worker=FlirWorker(flir_camera=DummyFLIRCamera()),
    is_sample_illumination_continuous=True
)


#
# # # Real
# brillouin_manager = BrillouinManager(
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


class BrillouinViewerMicroscope(QWidget):

    # Signals Outgoing
    gui_ready = pyqtSignal()
    apply_camera_settings_requested = pyqtSignal(dict)
    update_andor_config_requested = pyqtSignal(object)
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
    run_calibration_requested = pyqtSignal()
    take_measurement_requested = pyqtSignal(object)
    shutdown_requested = pyqtSignal()
    get_calibration_results_requested = pyqtSignal()
    toggle_do_live_fitting_requested = pyqtSignal()
    close_all_shutters_requested = pyqtSignal()

    # Flir camera
    request_flir_update_settings = pyqtSignal(object)

    # Microscope
    request_led_update_settings = pyqtSignal(object)


    def __init__(self):
        super().__init__()


        self.setWindowTitle("Brillouin Viewer (Live)")

        self.brillouin_signaller = BrillouinSignaller(manager=brillouin_backend)
        self.brillouin_signaller_thread = QThread()
        self.brillouin_signaller.moveToThread(self.brillouin_signaller_thread)

        # Sending signals
        self.apply_camera_settings_requested.connect(self.brillouin_signaller.apply_camera_settings)
        self.update_andor_config_requested.connect(self.brillouin_signaller.update_andor_config_settings)
        self.emit_camera_settings_requested.connect(self.brillouin_signaller.emit_camera_settings)
        self.toggle_camera_shutter_requested.connect(self.brillouin_signaller.toggle_camera_shutter)
        self.start_live_requested.connect(self.brillouin_signaller.start_live_view)
        self.stop_live_requested.connect(self.brillouin_signaller.stop_live_view)
        self.update_microwave_freq_requested.connect(self.brillouin_signaller.set_microwave_frequency)
        self.toggle_bg_subtraction_requested.connect(self.brillouin_signaller.toggle_background_subtraction)
        self.snap_requested.connect(self.brillouin_signaller.snap_and_fit)
        self.toggle_reference_mode_requested.connect(self.brillouin_signaller.toggle_reference_mode)
        self.acquire_background_requested.connect(self.brillouin_signaller.acquire_background_image)
        self.run_calibration_requested.connect(self.brillouin_signaller.run_calibration)
        self.take_measurement_requested.connect(self.brillouin_signaller.take_axial_scan)
        self.gui_ready.connect(self.brillouin_signaller.on_gui_ready)
        self.shutdown_requested.connect(self.brillouin_signaller.close)
        self.get_calibration_results_requested.connect(self.brillouin_signaller.get_calibration_results)
        self.toggle_do_live_fitting_requested.connect(self.brillouin_signaller.toggle_do_live_fitting)
        self.close_all_shutters_requested.connect(self.brillouin_signaller.close_all_shutters)

        # Flir camera settings
        self.request_flir_update_settings.connect(self.brillouin_signaller.flir_update_settings)

        # Microscope
        self.request_led_update_settings.connect(self.brillouin_signaller.update_microscope_leds)

        # Receiving signals
        self.brillouin_signaller.calibration_finished.connect(self.calibration_finished)
        self.brillouin_signaller.background_subtraction_state.connect(self.update_bg_subtraction)
        self.brillouin_signaller.background_available_state.connect(self.handle_is_bg_available)
        self.brillouin_signaller.reference_mode_state.connect(self.update_reference_ui)
        self.brillouin_signaller.camera_settings_ready.connect(self.populate_camera_ui)
        self.brillouin_signaller.camera_shutter_state_changed.connect(self.update_camera_shutter_button)
        self.brillouin_signaller.frame_and_fit_ready.connect(self.display_result, Qt.QueuedConnection)
        self.brillouin_signaller.measurement_result_ready.connect(self.handle_measurement_results)
        self.brillouin_signaller.zaber_lens_position_updated.connect(self.update_zaber_position)
        self.brillouin_signaller.microwave_frequency_updated.connect(self.update_ref_freq_input)
        self.brillouin_signaller.calibration_result_ready.connect(self.handle_requested_calibration)
        self.brillouin_signaller.do_live_fitting_state.connect(self.update_do_live_fitting_checkbox)
        self.brillouin_signaller.gui_ready_received.connect(self.brillouin_signaller.on_gui_ready)
        self.brillouin_signaller.update_system_state_in_frontend.connect(self.update_system_state_label)
        self.brillouin_signaller.flir_frame_ready.connect(self.display_flir_frame)

        # Connect signals BEFORE starting the thread
        self.brillouin_signaller.log_message.connect(lambda msg: print("[Signaller]", msg))
        self.brillouin_signaller_thread.started.connect(self.run_gui)

        # Start the thread after all connections
        self.brillouin_signaller_thread.start()

        self.init_ui()

        self.update_gui()


    def init_ui(self):
        outer_layout = QHBoxLayout()  # MAIN HORIZONTAL: Left (controls) | Right (plot + display)
        self.setLayout(outer_layout)

        # ---------------- LEFT COLUMN: Settings ----------------
        left_column_layout = QVBoxLayout()
        left_column_layout.addWidget(self.create_control_group())
        left_column_layout.addWidget(self.create_andor_camera_group())
        left_column_layout.addWidget(self.create_fitting_group())
        left_column_layout.addWidget(self.create_reference_group())
        left_column_layout.addWidget(self.create_background_group())
        left_column_layout.addWidget(self.create_objective_lens_group())
        left_column_layout.addWidget(self.create_flir_camera_group())
        left_column_layout.addWidget(self.create_leds_group())

        left_column_layout.addStretch()
        outer_layout.addLayout(left_column_layout, 0)

        # ---------------- RIGHT: Plots + Main Display ----------------
        right_layout = QVBoxLayout()

        plot_row_layout = QHBoxLayout()

        self.fig, (self.ax_img, self.ax_fit) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.subplots_adjust(hspace=0.9)
        self.canvas = FigureCanvas(self.fig)
        plot_row_layout.addWidget(self.create_andor_display_group())

        # --- FLIR Camera Display ---
        self.flir_group = QGroupBox("Flir Camera")
        self.flir_layout = QVBoxLayout()

        self.flir_image_label = QLabel("No FLIR image")
        self.flir_image_label.setFixedSize(320, 240)
        self.flir_image_label.setAlignment(Qt.AlignCenter)
        self.flir_image_label.setStyleSheet("background-color: black; color: white;")

        self.flir_layout.addWidget(self.flir_image_label)
        self.flir_group.setLayout(self.flir_layout)

        plot_row_layout.addWidget(self.flir_group)

        right_layout.addLayout(plot_row_layout)
        outer_layout.addLayout(right_layout, 1)


    def update_gui(self):
        # Update the gui
        self.brillouin_signaller.update_gui()




    # ---------------- UI Sections ---------------- #
    def create_control_group(self):
        group = QGroupBox("Control")
        layout = QHBoxLayout()

        stop_btn = QPushButton("STOP")
        stop_btn.clicked.connect(self.on_stop_clicked)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.on_cancel_event_clicked)

        restart_btn = QPushButton("Restart")
        restart_btn.clicked.connect(self.on_restart_clicked)

        self.state_label = QLabel("● IDLE")
        self.state_label.setStyleSheet("color: gray; font-weight: bold")

        layout.addWidget(stop_btn)
        layout.addWidget(cancel_btn)
        layout.addWidget(restart_btn)
        layout.addWidget(self.state_label)
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

    def create_objective_lens_group(self):
        group = QGroupBox("Objective Lens")
        layout = QFormLayout()

        # Working Distance (ComboBox)
        self.eff_focal_length_combo = QComboBox()
        self.eff_focal_length_combo.addItems([
            "165mm (Zeiss)",
            "180mm (Olympus)"
        ])
        layout.addRow("EFL:", self.eff_focal_length_combo)

        # Magnification (LineEdit for float/int input)
        self.magnification_combo = QComboBox()
        self.magnification_combo.addItems([
            "1X",
            "2X",
            "4X",
            "7.5X",
            "10X",
            "15X",
            "20X",
            "40X",
            "50X",
            "60X",
        ])
        layout.addRow("Magnification:", self.magnification_combo)

        group.setLayout(layout)
        return group

    def create_flir_camera_group(self):
        group = QGroupBox("FLIR Camera")
        layout = QVBoxLayout()
        group.setLayout(layout)

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.on_flir_config_clicked)
        layout.addWidget(settings_btn)

        return group



    # LED------------------------------------
    def create_leds_group(self):
        group = QGroupBox("LEDs")
        layout = QVBoxLayout()

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.on_led_settings_clicked)

        layout.addWidget(settings_btn)
        group.setLayout(layout)
        return group

    # Frames
    def create_andor_display_group(self):
        group = QGroupBox("Andor Frame and Fitting")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        group.setLayout(layout)
        group.setMinimumWidth(700)
        return group


    # ---------------- Signal Handles ---------------- #
    def update_system_state_label(self, state: SystemState):
        if state == SystemState.IDLE:
            self.state_label.setText("● IDLE")
            self.state_label.setStyleSheet("color: gray; font-weight: bold")
        elif state == SystemState.BUSY:
            self.state_label.setText("● BUSY")
            self.state_label.setStyleSheet("color: orange; font-weight: bold")
        elif state == SystemState.FREERUNNING:
            self.state_label.setText("● LIVE")
            self.state_label.setStyleSheet("color: green; font-weight: bold")

    def update_flir_camera_settings(self, flir_config: FLIRConfig):
        self.request_flir_update_settings.emit(flir_config)

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


    def on_flir_config_clicked(self):
        dialog = FLIRConfigDialog(flir_update_config=self.update_flir_camera_settings, parent=self)
        dialog.exec_()



    # ---------------- Toggle ---------------- #
    def toggle_background_subtraction(self):
        self.toggle_bg_subtraction_requested.emit()

    def toggle_illumination(self):
        self.toggle_illumination_requested.emit()

    def on_andor_configs_clicked(self):
        dialog = AndorConfigDialog(andor_update_config=self.update_andor_config_settings, parent=self)
        dialog.exec_()

    def update_andor_config_settings(self, andor_config: AndorConfig):
        self.update_andor_config_requested.emit(andor_config)

    def on_reference_configs_clicked(self):
        dialog = CalibrationConfigDialog(self)
        dialog.exec_()

    def on_fitting_configs_clicked(self):
        dialog = FindPeaksConfigDialog(self)
        dialog.exec_()

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

    @pyqtSlot(np.ndarray)
    def display_flir_frame(self, frame: np.ndarray):
        if frame is None:
            print("Frame is None")
            return

        frame = np.ascontiguousarray(frame)
        h, w = frame.shape
        bytes_per_line = w

        q_img = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)

        if q_img.isNull():
            print("QImage is null")
            return

        pixmap = QPixmap.fromImage(q_img)

        if pixmap.isNull():
            print("QPixmap is null")
            return

        self.flir_image_label.setPixmap(pixmap.scaled(
            self.flir_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    # ---------------- Handlers ---------------- #



    def update_camera_shutter_button(self, is_open: bool):
        text = "Close Shutter" if is_open else "Open Shutter"
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


    # -------------- Zaber Microscope -------
    def on_led_settings_clicked(self):
        dialog = LEDConfigDialog(update_led_config=self.update_led_settings, parent=self)
        dialog.exec_()

    def update_led_settings(self, led_config: LEDConfig):
        self.request_led_update_settings.emit(led_config)




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
    viewer = BrillouinViewerMicroscope()
    viewer.show()
    sys.exit(app.exec_())
