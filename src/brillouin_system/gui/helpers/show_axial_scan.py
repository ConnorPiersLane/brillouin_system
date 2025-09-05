import sys
import pickle
import numpy as np
import scipy

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QSpinBox, QFileDialog, QGroupBox, QPushButton
)
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import norm

from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.my_dataclasses.human_interface_measurements import (
    AxialScan, fit_axial_scan, analyze_axial_scan
)
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts


#
# plt.rcParams.update({
#     "font.size": 8,         # smaller base font
#     "axes.titlesize": 9,    # subplot titles
#     "axes.labelsize": 8,    # x/y labels
#     "xtick.labelsize": 7,   # x tick labels
#     "ytick.labelsize": 7,   # y tick labels
#     "legend.fontsize": 7,   # legend text
# })


class AxialScanViewer(QWidget):
    def __init__(self, axial_scan: AxialScan):
        super().__init__()

        self.axial_scan = axial_scan
        self.setWindowTitle(f"Axial Scan Viewer - ID: {axial_scan.id}")

        fitted_axial_scan_data = fit_axial_scan(axial_scan)
        self.analyzed_data = analyze_axial_scan(fitted_axial_scan_data)
        self.current_index = 0

        self.init_ui()
        self.print_scan_overview()
        self.update_display()

    def init_ui(self):
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        # --- Scan Info ---
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-weight: bold;")
        outer_layout.addWidget(self.info_label)

        # --- Plotting area ---
        plot_group = QGroupBox("Axial Scan Data")
        plot_layout = QVBoxLayout()

        # Create independent figure (no global plt state)
        self.fig = Figure(figsize=(8, 6))

        # Create subplots explicitly
        self.ax_img = self.fig.add_subplot(311)
        self.ax_spec = self.fig.add_subplot(312)
        self.ax_axial = self.fig.add_subplot(313)

        # Adjust layout
        self.fig.subplots_adjust(hspace=1, bottom=0.15)

        # Wrap in a Qt canvas
        self.canvas = FigureCanvas(self.fig)

        # Add to layout
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        outer_layout.addWidget(plot_group)

        # --- Navigation ---
        nav_layout = QHBoxLayout()

        self.left_btn = QPushButton("←")
        self.left_btn.clicked.connect(self.on_left_clicked)
        nav_layout.addWidget(self.left_btn)

        nav_layout.addWidget(QLabel("Measurement #:"))
        self.index_spinner = QSpinBox()
        self.index_spinner.setRange(0, len(self.axial_scan.measurements) - 1)
        self.index_spinner.valueChanged.connect(self.on_index_changed)
        nav_layout.addWidget(self.index_spinner)

        self.right_btn = QPushButton("→")
        self.right_btn.clicked.connect(self.on_right_clicked)
        nav_layout.addWidget(self.right_btn)

        self.analyze_btn = QPushButton("Analyze SNR")
        self.analyze_btn.clicked.connect(self.on_analyze_snr)
        nav_layout.addWidget(self.analyze_btn)

        nav_layout.addStretch()
        outer_layout.addLayout(nav_layout)

    def update_display(self):
        mp = self.axial_scan.measurements[self.current_index]
        self.print_measurement_info(self.current_index)
        frame = mp.frame_andor

        self.info_label.setText(
            f"ID: {self.axial_scan.id} | "
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

        # --- Spectrum + Fit (middle subplot) ---
        self.ax_spec.clear()

        fit = self.analyzed_data.fitted_scan.fitted_spectras[self.current_index]
        if fit is not None:
            self.ax_spec.plot(fit.x_pixels, fit.sline, 'k.', label="Spectrum")
            self.ax_spec.plot(fit.x_fit_refined, fit.y_fit_refined, 'r--', label="Fit")

        self.ax_spec.set_title(f"Spectrum at Z = {mp.lens_zaber_position:.2f} µm")
        self.ax_spec.set_xlabel("Pixel (X)")
        self.ax_spec.set_ylabel("Intensity (Σ Y)")
        self.ax_spec.legend()

        # --- Axial Scan Results (bottom subplot) ---
        self.ax_axial.clear()
        positions = [m.lens_zaber_position for m in self.axial_scan.measurements]

        config = calibration_config.get()

        if config.reference == 'left':
            freq_shifts = [fs.freq_shift_left_peak_ghz for fs in self.analyzed_data.freq_shifts]
        elif config.reference == 'right':
            freq_shifts = [fs.freq_shift_right_peak_ghz for fs in self.analyzed_data.freq_shifts]
        else:
            freq_shifts = [fs.freq_shift_peak_distance_ghz for fs in self.analyzed_data.freq_shifts]

        self.ax_axial.plot(range(len(freq_shifts)), freq_shifts, 'bo-', label="Frequency Shift")
        self.ax_axial.plot(self.current_index, freq_shifts[self.current_index], 'ro', markersize=10, label="Current")

        self.ax_axial.set_xticks(range(len(positions)))
        self.ax_axial.set_xticklabels([f"{i}\n{pos:.1f}" for i, pos in enumerate(positions)], rotation=45)
        self.ax_axial.set_xlabel("Index / Zaber Position (µm)")
        self.ax_axial.set_ylabel("Frequency Shift (GHz)")
        self.ax_axial.set_title("Axial Scan Result")
        self.ax_axial.legend()
        self.ax_axial.grid(True)

        self.canvas.draw()

    def on_index_changed(self, value):
        self.current_index = value
        self.update_display()

    def on_left_clicked(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.index_spinner.setValue(self.current_index)

    def on_right_clicked(self):
        if self.current_index < len(self.axial_scan.measurements) - 1:
            self.current_index += 1
            self.index_spinner.setValue(self.current_index)

    def on_analyze_snr(self):
        """Plot histogram of frequency shifts with Gaussian fit."""
        config = calibration_config.get()

        if config.reference == 'left':
            freq_shifts = [fs.freq_shift_left_peak_ghz for fs in self.analyzed_data.freq_shifts]
        elif config.reference == 'right':
            freq_shifts = [fs.freq_shift_right_peak_ghz for fs in self.analyzed_data.freq_shifts]
        else:
            freq_shifts = [fs.freq_shift_peak_distance_ghz for fs in self.analyzed_data.freq_shifts]

        freq_shifts = np.array(freq_shifts)

        # --- Compute stats ---
        mu = np.mean(freq_shifts)
        sigma = np.std(freq_shifts, ddof=1)  # sample std
        n = len(freq_shifts)

        print("==== Analyze SNR ====")
        print(f"Mean: {mu:.3f} GHz")
        print(f"Std: {sigma:.3f} GHz")
        print(f"n: {n}")
        print(f"Reference Peak (left, right, distance): {config.reference}")
        print("=====================")

        # --- Plot histogram ---
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        counts, bins, patches = ax.hist(freq_shifts, bins=15, color="skyblue", edgecolor="black", density=False,
                                        alpha=0.6)

        # Gaussian fit overlay
        x = np.linspace(min(freq_shifts), max(freq_shifts), 200)
        bin_width = bins[1] - bins[0]
        pdf = norm.pdf(x, mu, sigma) * n * bin_width
        ax.plot(x, pdf, "r-", lw=2, label=f"Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}")

        ax.set_xlabel("Frequency Shift (GHz)")
        ax.set_ylabel("Density")
        ax.set_title("Histogram of Frequency Shifts with Gaussian Fit")
        ax.legend()

        # --- Show histogram in a new window ---
        hist_window = QWidget()
        hist_window.setWindowTitle("Analyze SNR - Frequency Shift Histogram")
        layout = QVBoxLayout(hist_window)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        hist_window.setLayout(layout)
        hist_window.resize(600, 400)
        hist_window.show()

        # Keep reference
        if not hasattr(self, "open_hist_windows"):
            self.open_hist_windows = []
        self.open_hist_windows.append(hist_window)

    def print_scan_overview(self):
        """Print static information about the scan (once at init)."""
        scan = self.axial_scan
        print("==== Axial Scan Overview ====")
        print(f"Internal tracker i: {scan.i}")
        print(f"ID: {scan.id}")
        print(f"Number of measurements: {len(scan.measurements)}")

        if scan.calibration_params:
            print(f"Calibration Params: {scan.calibration_params}")
        else:
            print("Calibration Params: None")

        if scan.eye_location:
            print(f"Eye Location index: {scan.eye_location.index}")
        else:
            print("Eye Location: None")

        ss = scan.system_state
        print(f"System State: reference_mode={ss.is_reference_mode}, "
              f"bg_subtraction={ss.is_do_bg_subtraction_active}, "
              f"exposure={ss.andor_camera_info.exposure}, "
              f"gain={ss.andor_camera_info.gain}")
        print("=============================")

    def print_measurement_info(self, index: int):
        """Print info for the given measurement index (depends on current_index)."""

        def fmt(val, precision=3):
            return f"{val:.{precision}f}" if val is not None else "N/A"

        mp = self.axial_scan.measurements[index]
        freq_shift = self.analyzed_data.freq_shifts[index]
        photons: PhotonsCounts = self.analyzed_data.fitted_scan.fitted_photon_counts[index]

        print(f"--- Measurement {index} ---")
        print(f"Zaber position: {mp.lens_zaber_position:.2f} µm")
        print(
            f"Freq shifts: left={fmt(freq_shift.freq_shift_left_peak_ghz)}, "
            f"right={fmt(freq_shift.freq_shift_right_peak_ghz)}, "
            f"distance={fmt(freq_shift.freq_shift_peak_distance_ghz)}"
        )
        print(
            f"FWHM: left={fmt(freq_shift.fwhm_left_peak_ghz)}, "
            f"right={fmt(freq_shift.fwhm_right_peak_ghz)}"
        )
        print(
            f"Photons: left={fmt(photons.left_peak_photons)}, "
            f"right={fmt(photons.right_peak_photons)}, "
            f"total={fmt(photons.total_photons)}"
        )
        print("----------------------------")


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

    # some pickles may contain [AxialScan] list, so pick first
    if isinstance(scan, list):
        scan = scan[0]

    viewer = AxialScanViewer(scan)

    viewer.show()

    sys.exit(app.exec_())
