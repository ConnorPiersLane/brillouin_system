import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QSpinBox, QGroupBox, QPushButton, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import norm

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.human_interface_measurements import (
    AxialScan, fit_axial_scan, analyze_axial_scan, get_freq_shift
)
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts
from brillouin_system.spectrum_fitting.spectrum_analyzer import SpectrumAnalyzer, TheoreticalPeakStdError, \
    MeasuredStatistics


class AxialScanViewer(QWidget):
    """GUI for visualizing and analyzing axial scan data."""

    def __init__(self, axial_scan: AxialScan):
        super().__init__()
        self.axial_scan = axial_scan
        self.calc = CalibrationCalculator(self.axial_scan.calibration_params)
        self.setWindowTitle(f"Axial Scan Viewer - ID: {axial_scan.id}")

        # Analysis pipeline
        self.fitted_axial_scan_data = fit_axial_scan(axial_scan)
        self.analyzed_data = analyze_axial_scan(self.fitted_axial_scan_data)
        self.freq_shifts = [get_freq_shift(af) for af in self.analyzed_data.freq_shifts]

        # State
        self.current_index = 0
        self.open_hist_windows = []

        # Build UI
        self.init_ui()
        self.print_scan_overview()
        self.update_display()

    # ---------------- UI Setup ----------------

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.info_label)

        layout.addWidget(self.setup_plot_area())
        layout.addLayout(self.setup_navigation())

    def setup_plot_area(self) -> QGroupBox:
        group = QGroupBox("Axial Scan Data")
        vbox = QVBoxLayout()

        self.fig = Figure(figsize=(8, 6))
        self.ax_img = self.fig.add_subplot(311)
        self.ax_spec = self.fig.add_subplot(312)
        self.ax_axial = self.fig.add_subplot(313)
        self.fig.subplots_adjust(hspace=1)
        self.fig.subplots_adjust(bottom=0.2)  #

        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas)
        group.setLayout(vbox)
        return group

    def setup_navigation(self) -> QHBoxLayout:
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
        return nav_layout

    # ---------------- Plotting ----------------

    def update_display(self):
        mp = self.axial_scan.measurements[self.current_index]
        self.print_measurement_info(self.current_index)

        self.info_label.setText(
            f"ID: {self.axial_scan.id} | "
            f"Index: {self.current_index + 1} / {len(self.axial_scan.measurements)} | "
            f"Z pos: {mp.lens_zaber_position:.2f} µm"
        )

        self.plot_frame(mp.frame_andor)
        self.plot_spectrum(self.current_index)
        self.plot_axial_scan()

        self.canvas.draw()

    def plot_frame(self, frame: np.ndarray):
        self.ax_img.cla()
        self.ax_img.imshow(frame, cmap="gray", aspect="equal", interpolation="none", origin="upper")
        self.ax_img.set_title("Andor Frame")
        self.ax_img.set_xlabel("Pixel (X)")
        self.ax_img.set_ylabel("Pixel (Y)")

    def plot_spectrum(self, idx: int):
        self.ax_spec.cla()
        fit: FittedSpectrum = self.analyzed_data.fitted_scan.fitted_spectras[idx]
        self.ax_spec.plot(fit.x_pixels, fit.sline, 'k.', label="Spectrum")

        if fit.is_success:
            self.ax_spec.plot(fit.x_pixels[fit.mask_for_fitting], fit.sline[fit.mask_for_fitting], 'r.', label="Used")
            self.ax_spec.plot(fit.x_fit_refined, fit.y_fit_refined, 'r--', label="Fit")

        self.ax_spec.set_title(f"Spectrum at Z = {self.axial_scan.measurements[idx].lens_zaber_position:.2f} µm")
        self.ax_spec.set_xlabel("Pixel (X)")
        self.ax_spec.set_ylabel("Intensity (Σ Y)")
        self.ax_spec.legend()

    def plot_axial_scan(self):
        self.ax_axial.cla()
        positions = [
            m.lens_zaber_position - self.axial_scan.measurements[0].lens_zaber_position
            for m in self.axial_scan.measurements
        ]
        self.ax_axial.plot(range(len(self.freq_shifts)), self.freq_shifts, 'bo-', label="Frequency Shift")

        y_val = self.freq_shifts[self.current_index]
        if y_val is not None and np.isfinite(y_val):
            self.ax_axial.plot(self.current_index, y_val, 'ro', markersize=10, label="Current")
            self.ax_axial.set_title(f"Freq (GHz): {y_val:.3f}")

        self.ax_axial.set_xticks(range(len(positions)))
        self.ax_axial.set_xticklabels([f"{i}\n{pos:.1f}" for i, pos in enumerate(positions)], rotation=0)
        self.ax_axial.set_xlabel("Index / Zaber Position (µm)")
        self.ax_axial.set_ylabel("Frequency Shift (GHz)")


        step = max(1, len(positions) // 20)
        self.ax_axial.set_xticks(range(0, len(positions), step))
        self.ax_axial.set_xticklabels(
            [f"{i}\n{pos:.1f}" for i, pos in enumerate(positions)][::step],
            rotation=0, ha="right"
        )

        self.ax_axial.legend()
        self.ax_axial.grid(True)

    # ---------------- Navigation ----------------

    def on_index_changed(self, value: int):
        self.current_index = value
        self.update_display()

    def on_left_clicked(self):
        if self.current_index > 0:
            self.index_spinner.setValue(self.current_index - 1)

    def on_right_clicked(self):
        if self.current_index < len(self.axial_scan.measurements) - 1:
            self.index_spinner.setValue(self.current_index + 1)

    # ---------------- Analysis ----------------

    def on_analyze_snr(self):
        """Plot histogram of frequency shifts with Gaussian fit."""
        try:
            self.print_statistics()

            freq_shifts = np.array([fs for fs in self.freq_shifts if fs is not None], dtype=float)
            freq_shifts = freq_shifts[~np.isnan(freq_shifts)]
            if freq_shifts.size == 0:
                QMessageBox.warning(self, "No Data", "No valid frequency shifts to analyze.")
                return

            mu, sigma, n = np.mean(freq_shifts), np.std(freq_shifts, ddof=1), len(freq_shifts)

            print("==== Analyze SNR ====")
            print(f"Mean: {mu:.3f} GHz")
            print(f"Std: {sigma*1000:.3f} MHz")
            print(f"n: {n}")
            print(f"Reference Peak (left, right, distance, centroid, dc): {calibration_config.get().reference}")
            print("=====================")

            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            counts, bins, _ = ax.hist(freq_shifts, bins=15, color="skyblue", edgecolor="black", alpha=0.6)

            if sigma > 0:
                x = np.linspace(min(freq_shifts), max(freq_shifts), 200)
                bin_width = bins[1] - bins[0]
                pdf = norm.pdf(x, mu, sigma) * n * bin_width
                ax.plot(x, pdf, "r-", lw=2, label=f"Gaussian Fit\nμ={mu:.3f} GHz, σ={sigma*1000:.1f} MHz")
            ax.set_xlabel("Frequency Shift (GHz)")
            ax.set_ylabel("Count")
            ax.set_title("Histogram of Frequency Shifts with Gaussian Fit")
            ax.legend()

            hist_window = QWidget()
            hist_window.setWindowTitle("Analyze SNR - Frequency Shift Histogram")
            layout = QVBoxLayout(hist_window)
            layout.addWidget(FigureCanvas(fig))
            hist_window.setLayout(layout)
            hist_window.resize(600, 400)
            hist_window.show()

            self.open_hist_windows.append(hist_window)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze SNR:\n{e}")

    # ---------------- Printing ----------------

    def print_statistics(self):
        self.calc.print_all_models()
        analyzer = SpectrumAnalyzer(self.calc)

        fit = self.analyzed_data.fitted_scan.fitted_spectras[self.current_index]
        photons = self.analyzed_data.fitted_scan.fitted_photon_counts[self.current_index]
        tpse = analyzer.theoretical_precision(
            fs=fit,
            photons=photons,
            bg_frame_std=self.axial_scan.system_state.bg_image.std_image,
            preamp_gain=self.axial_scan.system_state.andor_camera_info.preamp_gain,
            emccd_gain=self.axial_scan.system_state.andor_camera_info.gain
        )
        if tpse is not None:
            self.print_theoretical_precision(tpse)

        ms = analyzer.measured_precision(self.analyzed_data.freq_shifts)
        if ms is not None:
            self.print_measured_precision(ms)

    def print_scan_overview(self):
        scan = self.axial_scan
        print("==== Axial Scan Overview ====")
        print(f"ID: {scan.id}, Internal tracker i: {scan.i}")
        print(f"Number of measurements: {len(scan.measurements)}")
        self.calc.print_current_model()
        ss = scan.system_state
        print(f"System State: reference_mode={ss.is_reference_mode}, "
              f"bg_subtraction={ss.is_do_bg_subtraction_active}, "
              f"exposure={ss.andor_camera_info.exposure}, "
              f"gain={ss.andor_camera_info.gain}")
        print("=============================")

    def print_measurement_info(self, idx: int):
        mp = self.axial_scan.measurements[idx]
        freq_shift = self.analyzed_data.freq_shifts[idx]
        photons: PhotonsCounts = self.analyzed_data.fitted_scan.fitted_photon_counts[idx]

        def fmt(val, precision=3): return f"{val:.{precision}f}" if val is not None else "N/A"

        print(f"--- Measurement {idx} ---")
        print(f"Zaber position: {mp.lens_zaber_position:.2f} µm")
        print(f"Freq shifts: left={fmt(freq_shift.freq_shift_left_peak_ghz)}, "
              f"right={fmt(freq_shift.freq_shift_right_peak_ghz)}, "
              f"distance={fmt(freq_shift.freq_shift_peak_distance_ghz)}")
        print(f"HWHM: left={fmt(freq_shift.hwhm_left_peak_ghz)}, "
              f"right={fmt(freq_shift.hwhm_right_peak_ghz)}")
        print(f"Photons: left={fmt(photons.left_peak_photons)}, "
              f"right={fmt(photons.right_peak_photons)}, "
              f"total={fmt(photons.total_photons)}")
        print("----------------------------")

    # --- New helpers ---
    @staticmethod
    def print_theoretical_precision(tpse: "TheoreticalPeakStdError"):
        fmt = AxialScanViewer._fmt
        print("==== Theoretical Peak Std Error ====")
        print(f"Left Peak: photons={fmt(tpse.left_peak_photons)}, "
              f"pixelation={fmt(tpse.left_peak_pixelation)}, "
              f"bg={fmt(tpse.left_peak_bg)}, "
              f"total={fmt(tpse.left_peak_total)}")
        print(f"Right Peak: photons={fmt(tpse.right_peak_photons)}, "
              f"pixelation={fmt(tpse.right_peak_pixelation)}, "
              f"bg={fmt(tpse.right_peak_bg)}, "
              f"total={fmt(tpse.right_peak_total)}")
        print("===================================")

    @staticmethod
    def print_measured_precision(ms: "MeasuredStatistics"):
        fmt = AxialScanViewer._fmt
        print("==== Measured Statistics ====")
        print(f"Freq shifts (MHz): "
              f"left={fmt(ms.freq_shift_left_peak_mhz_std)}, "
              f"right={fmt(ms.freq_shift_right_peak_mhz_std)}, "
              f"distance={fmt(ms.freq_shift_peak_distance_mhz_std)}, "
              f"dc={fmt(ms.freq_shift_dc_mhz_std)}, "
              f"centroid={fmt(ms.freq_shift_centroid_mhz_std)}")
        print(f"Cov(left,right) = {fmt(ms.freq_cov_left_right, unit=' MHz²')}, "
              f"Corr(left,right) = {fmt(ms.freq_corr_left_right)}")
        print("===============================")

    @staticmethod
    def _fmt(val, precision=4, unit=""):
        """Safe formatter for floats that may be None."""
        return f"{val:.{precision}g}{unit}" if val is not None else "N/A"
