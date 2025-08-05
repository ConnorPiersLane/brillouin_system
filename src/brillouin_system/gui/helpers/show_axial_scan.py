import sys
import os
import pickle
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QSpinBox, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.human_interface_measurements import AxialScan



class AxialScanViewer(QWidget):
    def __init__(self, axial_scan: AxialScan, calibration_calculator: CalibrationCalculator = None):
        super().__init__()

        self.setWindowTitle(f"Axial Scan Viewer - ID: {axial_scan.id}")
        self.axial_scan = axial_scan
        self.current_index = 0

        self.init_ui()
        self.update_display()

        if calibration_calculator is None:
            print('No Calibration available, cannot calculate Frequency shifts')
            self.analyzed_axial_scan = None
        else:
            # self.analyzed_axial_scan = analyze_axial_scan(scan=axial_scan,
            #                                          calibration_calculator=calibration_calculator)

    def init_ui(self):
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        # --- Scan Info ---
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-weight: bold;")
        outer_layout.addWidget(self.info_label)

        # --- Plotting area ---
        plot_group = QGroupBox("Andor Frame and Spectrum")
        plot_layout = QVBoxLayout()

        self.fig, (self.ax_img, self.ax_spec) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.subplots_adjust(hspace=0.8)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        outer_layout.addWidget(plot_group)

        # --- Axial Scan Widget Placeholder ---
        self.axial_scan_group = QGroupBox("Axial Scan")
        self.axial_scan_layout = QVBoxLayout()
        self.axial_scan_group.setLayout(self.axial_scan_layout)
        outer_layout.addWidget(self.axial_scan_group)

        # --- Navigation ---
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Measurement #:"))
        self.index_spinner = QSpinBox()
        self.index_spinner.setRange(0, len(self.axial_scan.measurements) - 1)
        self.index_spinner.valueChanged.connect(self.on_index_changed)
        nav_layout.addWidget(self.index_spinner)
        nav_layout.addStretch()
        outer_layout.addLayout(nav_layout)

    def update_display(self):
        mp = self.axial_scan.measurements[self.current_index]
        frame = mp.frame_andor

        self.info_label.setText(
            f"ID: {self.axial_scan.id} | "
            f"Power: {self.axial_scan.power_mW:.2f} mW | "
            f"Index: {self.current_index + 1} / {len(self.axial_scan.measurements)} | "
            f"Z pos: {mp.lens_zaber_position:.2f} µm"
        )

        # --- Frame Display (top subplot) ---
        self.ax_img.clear()
        self.ax_img.imshow(frame, cmap="gray", aspect="equal", interpolation="none", origin="upper")
        self.ax_img.set_xticks(np.arange(0, frame.shape[1], 10))
        self.ax_img.set_yticks(np.arange(0, frame.shape[0], 5))
        self.ax_img.set_xlabel("Pixel (X)")
        self.ax_img.set_ylabel("Pixel (Y)")
        self.ax_img.set_title("Andor Frame")

        # --- Spectrum Plot (bottom subplot) ---
        self.ax_spec.clear()
        spectrum = np.sum(frame, axis=0)
        self.ax_spec.plot(spectrum, 'k.', label="Spectrum")
        self.ax_spec.set_title(f"Spectrum at Z = {mp.lens_zaber_position:.2f} µm")
        self.ax_spec.set_xlabel("Pixel (X)")
        self.ax_spec.set_ylabel("Intensity (Σ Y)")

        self.canvas.draw()

    def on_index_changed(self, value):
        self.current_index = value
        self.update_display()


def load_axial_scan_from_file(path: str) -> AxialScan:
    with open(path, "rb") as f:
        scan = pickle.load(f)
    return scan


if __name__ == "__main__":
    app = QApplication(sys.argv)

    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_path, _ = file_dialog.getOpenFileName(
        None, "Open Axial Scan", "", "Pickle Files (*.pkl);;All Files (*)"
    )

    if not file_path:
        sys.exit(0)

    scan = load_axial_scan_from_file(file_path)
    viewer = AxialScanViewer(scan[0])
    viewer.resize(800, 700)
    viewer.show()

    sys.exit(app.exec_())
