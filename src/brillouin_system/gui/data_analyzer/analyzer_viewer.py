# analyzer_viewer.py

import sys
import pickle
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QListWidget, QRadioButton, QButtonGroup, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from brillouin_system.gui.data_analyzer.analyzer_manager import AnalyzerManager
from brillouin_system.gui.brillouin_viewer.config_dialog import ConfigDialog

from brillouin_system.my_dataclasses.calibration import (
    CalibrationCalculator,
    render_calibration_to_pixmap,
    CalibrationImageDialog, calibrate,
)
from brillouin_system.config.config import calibration_config


class AnalyzerViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyzer Viewer")
        self.setMinimumSize(1200, 720)
        self.analyze_manager = AnalyzerManager()
        self.init_ui()

    def init_ui(self):
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        # --------- Row 1: Display Area ---------
        display_row = QHBoxLayout()

        # Left: Frame + Sline
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

        # Right: Mako image
        self.mako_label = QLabel("Display area")
        self.mako_label.setAlignment(Qt.AlignCenter)
        self.mako_label.setStyleSheet("background-color: black; color: white;")
        self.mako_label.setFixedSize(576, 432)

        display_row.addLayout(left_display_col)
        display_row.addWidget(self.mako_label)
        outer_layout.addLayout(display_row)

        # --------- Row 2: Bottom Controls ---------
        bottom_row = QHBoxLayout()

        # --- Column 1: Series Control Box ---
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

        # --- Column 2: Calibration & Fitting Box ---
        calibration_group = QGroupBox("Calibration and Fitting")
        calibration_layout = QVBoxLayout()

        self.radio_use_series = QRadioButton("Use Series Calibration")
        self.radio_use_external = QRadioButton("Use External Calibration")
        self.radio_use_series.setChecked(True)

        self.calibration_button_group = QButtonGroup()
        self.calibration_button_group.addButton(self.radio_use_series)
        self.calibration_button_group.addButton(self.radio_use_external)

        self.load_calibration_btn = QPushButton("Load Calibration")
        self.load_calibration_btn.clicked.connect(self.load_calibration_file)

        self.loaded_calibration_label = QLabel("Loaded Calibration: None")

        self.show_calibration_btn = QPushButton("Show Calibration")
        self.show_calibration_btn.clicked.connect(self.show_calibration)
        self.show_calibration_btn.setEnabled(False)

        self.radio_no_bg = QRadioButton("No BG Subtraction")
        self.radio_with_bg = QRadioButton("With BG Subtraction")
        self.radio_no_bg.setChecked(True)

        self.bg_button_group = QButtonGroup()
        self.bg_button_group.addButton(self.radio_no_bg)
        self.bg_button_group.addButton(self.radio_with_bg)

        self.config_series_btn = QPushButton("Config")
        self.config_series_btn.clicked.connect(self.open_config_dialog)

        # Add widgets to layout in logical order
        calibration_layout.addWidget(self.radio_use_series)
        calibration_layout.addWidget(self.radio_use_external)
        calibration_layout.addWidget(self.load_calibration_btn)
        calibration_layout.addWidget(self.loaded_calibration_label)
        calibration_layout.addWidget(self.show_calibration_btn)
        calibration_layout.addWidget(self.radio_no_bg)
        calibration_layout.addWidget(self.radio_with_bg)
        calibration_layout.addWidget(self.config_series_btn)

        calibration_group.setLayout(calibration_layout)
        bottom_row.addWidget(calibration_group)

        bottom_row.addStretch()
        outer_layout.addLayout(bottom_row)

        # Connect buttons
        self.add_series_btn.clicked.connect(self.load_series_file)
        self.remove_series_btn.clicked.connect(self.remove_selected_series)

    def load_calibration_file(self):
        path = self.analyze_manager.load_calibration_from_file()
        if path:
            filename = path.split("/")[-1]
            self.loaded_calibration_label.setText(f"Loaded Calibration: {filename}")
            self.show_calibration_btn.setEnabled(True)

    def show_calibration(self):
        use_series = self.radio_use_series.isChecked()
        calibration_data = None

        if use_series:
            selected_items = self.series_list_widget.selectedItems()
            if len(selected_items) != 1:
                QMessageBox.warning(
                    self,
                    "Invalid Selection",
                    "Please select exactly one series to use its calibration."
                )
                return

            index = self.series_list_widget.row(selected_items[0])
            if 0 <= index < len(self.analyze_manager.stored_measurement_series):
                calibration_data = self.analyze_manager.stored_measurement_series[index].calibration_data
            else:
                print("[AnalyzerViewer] Invalid series index.")
                return
        else:
            calibration_data = self.analyze_manager.external_calibration

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

    def open_config_dialog(self):
        dialog = ConfigDialog(self)
        dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = AnalyzerViewer()
    viewer.show()
    sys.exit(app.exec_())
