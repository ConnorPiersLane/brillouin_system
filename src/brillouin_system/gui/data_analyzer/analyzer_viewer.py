import sys
import pickle
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QListWidget, QRadioButton, QButtonGroup, QFileDialog,
    QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from brillouin_system.gui.data_analyzer.analyzer_manager import AnalyzerManager
from brillouin_system.gui.brillouin_viewer.config_dialog import ConfigDialog
from brillouin_system.my_dataclasses.analyzer_results import AnalyzedFrame
from brillouin_system.my_dataclasses.fitted_results import DisplayResults
from brillouin_system.my_dataclasses.calibration import (
    CalibrationCalculator, render_calibration_to_pixmap,
    CalibrationImageDialog, calibrate,
)
from brillouin_system.config.config import calibration_config


class AnalyzerViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyzer Viewer")
        self.setMinimumSize(1200, 720)
        self.analyze_manager = AnalyzerManager()

        self.current_measurement_index = 0
        self.current_fitted_results = []
        self.current_series = None
        self.analyzed_series_lookup = {}

        self.init_ui()

    def init_ui(self):
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        # Top display: frame + spectrum
        display_row = QHBoxLayout()
        left_display_col = QVBoxLayout()

        self.frame_label = QLabel("No Frame")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("background-color: black; color: white;")
        self.frame_label.setFixedSize(600, 150)

        self.fig, self.ax = plt.subplots(figsize=(6, 2.5))
        self.ax.set_title("Spectrum Fit | Interpeak: - px / - GHz")
        self.ax.set_xlabel("Pixel (X)")
        self.ax.set_ylabel("Intensity")
        self.fig.tight_layout()
        self.canvas = FigureCanvas(self.fig)

        left_display_col.addWidget(self.frame_label)
        left_display_col.addWidget(self.canvas)

        self.mako_label = QLabel("Display area")
        self.mako_label.setAlignment(Qt.AlignCenter)
        self.mako_label.setStyleSheet("background-color: black; color: white;")
        self.mako_label.setFixedSize(576, 432)

        display_row.addLayout(left_display_col)
        display_row.addWidget(self.mako_label)
        outer_layout.addLayout(display_row)

        # Bottom control layout
        bottom_row = QHBoxLayout()

        # --- Series Control ---
        series_group = QGroupBox("Series Controls")
        series_layout = QVBoxLayout()

        self.series_list_widget = QListWidget()
        self.series_list_widget.setSelectionMode(QListWidget.MultiSelection)

        self.add_series_btn = QPushButton("Add Series")
        self.remove_series_btn = QPushButton("Remove Selected")

        series_layout.addWidget(self.series_list_widget)
        series_layout.addWidget(self.add_series_btn)
        series_layout.addWidget(self.remove_series_btn)
        series_group.setLayout(series_layout)
        bottom_row.addWidget(series_group)

        # --- Calibration & Fitting ---
        calibration_group = QGroupBox("Calibration and Fitting")
        calibration_layout = QVBoxLayout()

        self.radio_use_calibration_in_series = QRadioButton("Use Series Calibration")
        self.radio_use_external = QRadioButton("Use External Calibration")
        self.radio_use_calibration_in_series.setChecked(True)

        self.calibration_button_group = QButtonGroup()
        self.calibration_button_group.addButton(self.radio_use_calibration_in_series)
        self.calibration_button_group.addButton(self.radio_use_external)

        self.load_calibration_btn = QPushButton("Load Calibration")
        self.load_calibration_btn.clicked.connect(self.load_calibration_file)

        self.loaded_calibration_label = QLabel("Loaded Calibration: None")

        self.show_calibration_btn = QPushButton("Show Calibration")
        self.show_calibration_btn.clicked.connect(self.show_calibration)

        self.config_series_btn = QPushButton("Config")
        self.config_series_btn.clicked.connect(self.open_config_dialog)

        self.bg_sub_checkbox = QCheckBox("Do BG Subtraction")
        self.bg_sub_checkbox.setChecked(False)
        self.bg_sub_checkbox.setToolTip("Enable background subtraction before spectrum fitting.")

        self.analyze_series_btn = QPushButton("Analyze Selected Series")
        self.analyze_series_btn.clicked.connect(self.analyze_selected_series)

        calibration_layout.addWidget(self.radio_use_calibration_in_series)
        calibration_layout.addWidget(self.radio_use_external)
        calibration_layout.addWidget(self.load_calibration_btn)
        calibration_layout.addWidget(self.loaded_calibration_label)
        calibration_layout.addWidget(self.show_calibration_btn)
        calibration_layout.addWidget(self.config_series_btn)
        calibration_layout.addWidget(self.bg_sub_checkbox)
        calibration_layout.addWidget(self.analyze_series_btn)

        calibration_group.setLayout(calibration_layout)
        bottom_row.addWidget(calibration_group)

        # --- One Series Navigation ---
        self.single_series_group = QGroupBox("One Measurement Series")
        single_series_layout = QVBoxLayout()

        self.series_status_label = QLabel("Show Series: No series selected.")
        single_series_layout.addWidget(self.series_status_label)

        nav_layout = QHBoxLayout()
        self.left_btn = QPushButton("Left")
        self.right_btn = QPushButton("Right")
        self.index_label = QLabel("0 of 0")
        nav_layout.addWidget(self.left_btn)
        nav_layout.addWidget(self.right_btn)
        nav_layout.addWidget(self.index_label)
        single_series_layout.addLayout(nav_layout)

        self.left_btn.clicked.connect(self.go_left)
        self.right_btn.clicked.connect(self.go_right)

        self.single_series_group.setLayout(single_series_layout)
        bottom_row.addWidget(self.single_series_group)

        bottom_row.addStretch()
        outer_layout.addLayout(bottom_row)

        self.add_series_btn.clicked.connect(self.load_series_file)
        self.remove_series_btn.clicked.connect(self.remove_selected_series)

    def open_config_dialog(self):
        dialog = ConfigDialog(self)
        dialog.exec_()

    def load_calibration_file(self):
        path = self.analyze_manager.load_calibration_from_file()
        if path:
            filename = path.split("/")[-1]
            self.loaded_calibration_label.setText(f"Loaded Calibration: {filename}")
            self.show_calibration_btn.setEnabled(True)

    def show_calibration(self):
        use_series = self.radio_use_calibration_in_series.isChecked()

        if use_series:
            selected_items = self.series_list_widget.selectedItems()
            if len(selected_items) != 1:
                QMessageBox.warning(self, "Invalid Selection", "Please select exactly one series.")
                return

            index = self.series_list_widget.row(selected_items[0])
            series = self.analyze_manager.stored_measurement_series[index]
            calibration_data = series.calibration_data
        else:
            calibration_data = self.analyze_manager.calibration_data_from_file

        if calibration_data is None:
            QMessageBox.warning(self, "No Calibration", "No calibration data is available.")
            return

        try:
            calculator = CalibrationCalculator(calibrate(calibration_data))
            reference = calibration_config.get().reference
            pixmap = render_calibration_to_pixmap(calibration_data, calculator, reference)
            dialog = CalibrationImageDialog(pixmap, parent=self)
            dialog.exec_()
        except Exception as e:
            print(f"[AnalyzerViewer] Failed to display calibration: {e}")

    def load_series_file(self):
        series_info_list = self.analyze_manager.load_measurements_from_file()
        for info in series_info_list:
            self.series_list_widget.addItem(info)

    def remove_selected_series(self):
        selected_items = self.series_list_widget.selectedItems()
        for item in selected_items:
            row = self.series_list_widget.row(item)
            self.series_list_widget.takeItem(row)
            self.analyze_manager.remove_measurement_series(row)

    def analyze_selected_series(self):
        selected_items = self.series_list_widget.selectedItems()
        if len(selected_items) != 1:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly one series.")
            self.series_status_label.setText("Show Series: Please select exactly one series.")
            return

        index = self.series_list_widget.row(selected_items[0])
        indices = [index]
        use_series_calibration = self.radio_use_calibration_in_series.isChecked()
        do_bg_sub = self.bg_sub_checkbox.isChecked()

        self.analyze_manager.analyze_selected_series(
            indices=indices,
            is_do_bg_subtraction=do_bg_sub,
            is_use_own_calibration_data=use_series_calibration
        )

        self.current_series = self.analyze_manager.stored_measurement_series[index]
        self.current_fitted_results = self.analyze_manager.analyzed_series_lookup[index]
        self.current_measurement_index = 0
        self.series_status_label.setText("Show Series: OK.")
        self.update_series_display()

    def display_result(self, analyzed_frame: AnalyzedFrame):
        frame = analyzed_frame.frame
        x_px = analyzed_frame.fitted_spectrum.x_pixels
        spectrum = analyzed_frame.fitted_spectrum.sline

        self.ax.clear()
        self.ax.plot(x_px, spectrum, 'k.', label="Spectrum")

        interpeak = None
        freq_shift_ghz = None

        if analyzed_frame.fitted_spectrum.is_success:
            x_fit_refined = analyzed_frame.fitted_spectrum.x_fit_refined
            y_fit_refined = analyzed_frame.fitted_spectrum.y_fit_refined
            interpeak = analyzed_frame.fitted_spectrum.inter_peak_distance
            freq_shift_ghz = analyzed_frame.freq_shift_peak_distance_ghz
            self.ax.plot(x_fit_refined, y_fit_refined, 'r--', label="Fit")
            self.ax.legend()

        if interpeak is not None and freq_shift_ghz is not None:
            title = f"Spectrum Fit | Interpeak: {interpeak:.2f} px / {freq_shift_ghz:.3f} GHz"
        elif interpeak is not None:
            title = f"Spectrum Fit | Interpeak: {interpeak:.2f} px / - GHz"
        elif freq_shift_ghz is not None:
            title = f"Spectrum Fit | Interpeak: - px / {freq_shift_ghz:.3f} GHz"
        else:
            title = "Spectrum Fit | Interpeak: - px / - GHz"

        self.ax.set_title(title)
        self.ax.set_xlabel("Pixel (X)")
        self.ax.set_ylabel("Intensity")
        self.canvas.draw()

    def go_left(self):
        if self.current_measurement_index > 0:
            self.current_measurement_index -= 1
            self.update_series_display()

    def go_right(self):
        if self.current_series and self.current_measurement_index < len(self.current_series.measurements) - 1:
            self.current_measurement_index += 1
            self.update_series_display()

    def update_series_display(self):
        if not self.current_fitted_results:
            self.index_label.setText("0 of 0")
            return
        self.index_label.setText(f"{self.current_measurement_index + 1} of {len(self.current_fitted_results)}")
        self.display_result(self.current_fitted_results[self.current_measurement_index])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = AnalyzerViewer()
    viewer.show()
    sys.exit(app.exec_())
