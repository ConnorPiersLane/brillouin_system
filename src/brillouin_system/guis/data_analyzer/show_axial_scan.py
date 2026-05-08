

import pprint
from dataclasses import asdict

import numpy as np

from tifffile import imwrite


from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QSpinBox, QGroupBox, QPushButton, QMessageBox, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.stats import norm

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.calibration.config.calibration_config import CalibrationConfig, calibration_config
from brillouin_system.guis.data_analyzer.excel_export_axial_scan import (
    get_excel_row_data,
    BrillouinExport,
    load_from_excel,
    export_to_excel,
)
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.human_interface_measurements import (
    AxialScan, fit_axial_scan, AnalyzedSpectrum
)
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts
from brillouin_system.spectrum_fitting.spectrum_analyzer import TheoreticalPeakStdError, \
    AnalyzedFreqShifts, MeasuredStatistics, analyze_statistics


def fmt(val, precision=3): return f"{val:.{precision}f}" if val is not None else "N/A"


class AxialScanViewer(QWidget):
    """GUI for visualizing and analyzing axial scan data."""

    def __init__(self, axial_scan: AxialScan):
        super().__init__()
        self.axial_scan: AxialScan = axial_scan
        self.calc: CalibrationCalculator = CalibrationCalculator(self.axial_scan.calibration_params)

        self.setWindowTitle(f"Axial Scan Viewer - ID: {axial_scan.id}")

        # Get mode and peak side:
        cal_config: CalibrationConfig = calibration_config.get()
        self.peak_reference = cal_config.reference
        self.fitting_mode = cal_config.mode

        # Analysis pipeline
        self.list_analyzed_spectras: list[AnalyzedSpectrum] = fit_axial_scan(axial_scan)
        self.freq_shifts: list[float] = [self.get_freq(a.analyzed_shifts) for a in self.list_analyzed_spectras]
        # State

        self.current_index = 0
        self.open_hist_windows = []

        # Build UI
        self.init_ui()
        self.print_scan_overview()
        self.update_display()


    # ---------------- UI Setup ----------------
    def get_freq(self, analyzed_shifts: AnalyzedFreqShifts):

        if self.peak_reference == "left":
            return analyzed_shifts.freq_shift_left_peak_ghz


        elif self.peak_reference == "right":
            return analyzed_shifts.freq_shift_right_peak_ghz


        elif self.peak_reference == "distance":
            return analyzed_shifts.freq_shift_peak_distance_ghz

        else:
            raise ValueError(
                f"Unknown reference '{self.peak_reference}'. Use 'left', 'right', or 'distance'."
            )



    def init_ui(self):
        layout = QVBoxLayout(self)
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.info_label)

        layout.addWidget(self.setup_plot_area())

        controls = QVBoxLayout()
        controls.addLayout(self.setup_navigation())
        controls.addLayout(self.setup_second_row_controls())
        layout.addLayout(controls)

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

        self.save_new_btn = QPushButton("Save New Excel")
        self.save_new_btn.clicked.connect(self.save_new_excel)
        nav_layout.addWidget(self.save_new_btn)

        self.add_existing_btn = QPushButton("Add to Existing Excel")
        self.add_existing_btn.clicked.connect(self.add_to_existing_excel)
        nav_layout.addWidget(self.add_existing_btn)

        self.export_frame_btn = QPushButton("Export Frame (uint8)")
        self.export_frame_btn.clicked.connect(self.export_current_frame)
        nav_layout.addWidget(self.export_frame_btn)

        nav_layout.addStretch()
        return nav_layout

    def setup_second_row_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.show_cal_plot_btn = QPushButton("Show Calibration Plot")
        self.show_cal_plot_btn.clicked.connect(self.on_show_calibration_plot)
        row.addWidget(self.show_cal_plot_btn)

        row.addStretch()
        return row
    # ---------------- Plotting ----------------

    def update_display(self):
        mp = self.axial_scan.measurements[self.current_index]
        self.print_measurement_info()

        self.info_label.setText(
            f"ID: {self.axial_scan.id} | "
            f"Index: {self.current_index + 1} / {len(self.axial_scan.measurements)} | "
            f"Z pos: {mp.lens_zaber_position:.2f} µm"
        )


        if self.axial_scan.system_state.is_do_bg_subtraction_active:
            frame = mp.frame_andor - self.axial_scan.system_state.bg_image.median_image
        else:
            frame =mp.frame_andor
        self.plot_frame(frame)
        self.plot_spectrum()
        self.plot_axial_scan()

        self.canvas.draw()


        fitted_spectrum = self.list_analyzed_spectras[self.current_index].fitted_spectrum


    def plot_frame(self, frame: np.ndarray):
        self.ax_img.cla()
        self.ax_img.imshow(frame, cmap="gray", aspect="equal", interpolation="none", origin="upper")
        self.ax_img.set_title("Andor Frame")
        self.ax_img.set_xlabel("Pixel (X)")
        self.ax_img.set_ylabel("Pixel (Y)")

    def plot_spectrum(self):
        self.ax_spec.cla()
        fit: FittedSpectrum = self.list_analyzed_spectras[self.current_index].fitted_spectrum
        self.ax_spec.plot(fit.x_pixels, fit.sline, 'k.', label="Spectrum")

        if fit.is_success:
            self.ax_spec.plot(fit.x_pixels[fit.mask_for_fitting], fit.sline[fit.mask_for_fitting], 'r.', label="Used")
            self.ax_spec.plot(fit.x_fit_refined, fit.y_fit_refined, 'r--', label="Fit")

        self.ax_spec.set_title(f"Spectrum at Z = {round(self.axial_scan.measurements[self.current_index].lens_zaber_position)} µm")
        self.ax_spec.set_xlabel("Pixel (X)")
        self.ax_spec.set_ylabel("Intensity (Σ Y)")
        self.ax_spec.legend()

    def plot_axial_scan(self):
        self.ax_axial.cla()

        # Plot frequency shift vs index
        x = range(len(self.list_analyzed_spectras))
        self.ax_axial.plot(x, self.freq_shifts, 'bo-', label="Frequency Shift")

        # Highlight current point
        y_val = self.freq_shifts[self.current_index]
        if y_val is not None and np.isfinite(y_val):
            self.ax_axial.plot(self.current_index, y_val, 'ro', markersize=10, label="Current")
            self.ax_axial.set_title(f"Freq (GHz): {y_val:.3f}")

        # Set x-axis ticks (index only, downsampled)
        step = max(1, len(self.list_analyzed_spectras) // 20)
        self.ax_axial.set_xticks(range(0, len(self.list_analyzed_spectras), step))

        self.ax_axial.set_xlabel("Index")
        self.ax_axial.set_ylabel("Frequency Shift (GHz)")

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
            print("==== Analyze SNR ====")
            shifts: list[AnalyzedFreqShifts] = [s.analyzed_shifts for s in self.list_analyzed_spectras]
            stats: MeasuredStatistics = analyze_statistics(shifts)
            pprint.pprint(asdict(stats))

            freq_shifts = np.array([fs for fs in self.freq_shifts if fs is not None], dtype=float)
            freq_shifts = freq_shifts[~np.isnan(freq_shifts)]
            if freq_shifts.size == 0:
                QMessageBox.warning(self, "No Data", "No valid frequency shifts to analyze.")
                return

            mu, sigma, n = np.mean(freq_shifts), np.std(freq_shifts, ddof=1), len(freq_shifts)


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
    def print_calibration_models(self):
        self.calc.print_all_models()


    def print_scan_overview(self):
        scan = self.axial_scan
        print("==== Axial Scan Overview ====")
        print(f"ID: {scan.id}, Internal tracker i: {scan.i}")
        print(f"Number of measurements: {len(scan.measurements)}")

        self.calc.print_all_models()
        ss = scan.system_state
        print(f"System State: reference_mode={ss.is_reference_mode}, "
              f"bg_subtraction={ss.is_do_bg_subtraction_active}, "
              f"exposure={ss.andor_camera_info.exposure}, "
              f"gain={ss.andor_camera_info.gain}")
        if scan.reflection_result_forwards is not None:
            if scan.reflection_result_forwards.event_z_um is not None:
                print(f'Plane (forwards): {round(scan.reflection_result_forwards.event_z_um)}')
            else:
                print(f'Plane (forwards): None')
        if scan.reflection_result_backwards is not None:
            if scan.reflection_result_backwards.event_z_um is not None:
                print(f'Plane (backwards): {round(scan.reflection_result_backwards.event_z_um)}')
            else:
                print(f'Plane (backwards): None')

        if scan.eye_tracker_results is not None:
            if scan.eye_tracker_results.laser_position is not None:
                er = scan.eye_tracker_results
                print("Laser Position [mm]:")
                print(f"X: {fmt(er.laser_position[0], precision=2)}")
                print(f"Y: {fmt(er.laser_position[1], precision=2)}")
                print(f"Z: {fmt(er.laser_position[2], precision=2)}")
            else:
                print(f"Eye Tracker Position is None")
        print("=============================")

    def print_measurement_info(self):
        mp = self.axial_scan.measurements[self.current_index]
        freq_shift = self.list_analyzed_spectras[self.current_index].analyzed_shifts
        photons: PhotonsCounts = self.list_analyzed_spectras[self.current_index].photons

        print(f"--- Measurement {self.current_index} ---")
        print(f"Zaber position: {fmt(mp.lens_zaber_position, precision=0)} µm")
        print(f"Freq shifts poly: left={fmt(freq_shift.freq_shift_left_peak_ghz)}, "
              f"right={fmt(freq_shift.freq_shift_right_peak_ghz)}, "
              f"distance={fmt(freq_shift.freq_shift_peak_distance_ghz)}")
        print(f"HWHM (GHz): left={fmt(freq_shift.hwhm_left_peak_ghz)}, "
              f"right={fmt(freq_shift.hwhm_right_peak_ghz)}")
        print(f"Photons: left={fmt(photons.left_peak_photons, precision=0)}, "
              f"right={fmt(photons.right_peak_photons, precision=0)}, "
              f"total={fmt(photons.total_photons, precision=0)}")
        print("----------------------------")

    # --- New helpers ---
    @staticmethod
    def print_tpse(tpse: "TheoreticalPeakStdError"):
        print("==== Theoretical Peak Std Error (MHz) ====")
        print(f"Left Peak: photons={fmt(tpse.left_peak_photons_mhz, precision=0)}, "
              f"pixelation={fmt(tpse.left_peak_pixelation_mhz, precision=0)}, "
              f"bg={fmt(tpse.left_peak_bg_mhz, precision=0)}, "
              f"total={fmt(tpse.left_peak_total_mhz, precision=0)}")
        print(f"Right Peak: photons={fmt(tpse.right_peak_photons_mhz, precision=0)}, "
              f"pixelation={fmt(tpse.right_peak_pixelation_mhz, precision=0)}, "
              f"bg={fmt(tpse.right_peak_bg_mhz, precision=0)}, "
              f"total={fmt(tpse.right_peak_total_mhz, precision=0)}")
        print("===================================")


    # --- Export to Excel ---
    def _safe_sheet_name(self, name: str) -> str:
        invalid = ['\\', '/', '*', '?', ':', '[', ']']
        for ch in invalid:
            name = name.replace(ch, "_")
        return name[:31] or "Sheet1"

    def _build_export_rows(self) -> list[BrillouinExport]:
        rows: list[BrillouinExport] = []

        for i in range(len(self.list_analyzed_spectras)):
            row = get_excel_row_data(
                axial_scan=self.axial_scan, analyzed_spectrum=self.list_analyzed_spectras[i], idx=i,
            )

            rows.append(row)

        return rows

    def _get_default_excel_name(self) -> str:
        return f"{self.axial_scan.id}_brillouin_export.xlsx"

    def _write_rows_to_excel(self, file_path: str, rows: list[BrillouinExport]):
        if not file_path.lower().endswith(".xlsx"):
            file_path += ".xlsx"

        export_to_excel(rows, file_path)
        return file_path

    def save_new_excel(self):
        try:
            rows = self._build_export_rows()
            if not rows:
                QMessageBox.warning(self, "No Data", "There is no data to export.")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save New Excel",
                self._get_default_excel_name(),
                "Excel Files (*.xlsx)"
            )

            if not file_path:
                return

            file_path = self._write_rows_to_excel(file_path, rows)

            QMessageBox.information(
                self,
                "Excel Saved",
                f"Saved {len(rows)} rows to new file:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save Excel file:\n{e}")

    def add_to_existing_excel(self):
        try:
            new_rows = self._build_export_rows()
            if not new_rows:
                QMessageBox.warning(self, "No Data", "There is no data to export.")
                return

            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Add to Existing Excel",
                "",
                "Excel Files (*.xlsx)"
            )

            if not file_path:
                return

            existing_rows = load_from_excel(file_path, sheet_name=0)
            combined_rows = existing_rows + new_rows

            self._write_rows_to_excel(file_path, combined_rows)

            QMessageBox.information(
                self,
                "Excel Updated",
                f"Added {len(new_rows)} rows.\n"
                f"Workbook now contains {len(combined_rows)} rows:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Append Failed", f"Could not update Excel file:\n{e}")

    def export_current_frame(self):
        try:
            mp = self.axial_scan.measurements[self.current_index]

            if self.axial_scan.system_state.is_do_bg_subtraction_active:
                frame = mp.frame_andor - self.axial_scan.system_state.bg_image.median_image
            else:
                frame = mp.frame_andor

            # Normalize → uint8
            frame = np.array(frame, dtype=float)

            # Avoid division by zero
            if frame.max() > frame.min():
                norm = (frame - frame.min()) / (frame.max() - frame.min())
            else:
                norm = np.zeros_like(frame)

            frame_uint8 = (norm * 255).astype(np.uint8)

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Frame",
                f"frame_{self.current_index}.tiff",
                "TIFF (*.tiff *.tif);;PNG (*.png)"
            )

            if not file_path:
                return

            if file_path.lower().endswith((".tif", ".tiff")):
                imwrite(file_path, frame_uint8)

            elif file_path.lower().endswith(".png"):
                from PIL import Image
                Image.fromarray(frame_uint8).save(file_path)

            else:
                file_path += ".tiff"
                imwrite(file_path, frame_uint8)

            QMessageBox.information(self, "Saved", f"Saved to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{e}")

    def on_show_calibration_plot(self):
        """
        Show calibration plot for current reference:
          reference == "left"     -> left peak px vs GHz
          reference == "right"    -> right peak px vs GHz
          reference == "distance" -> inter-peak distance px vs GHz

        Also plots:
          - corresponding interpolation/reference points
          - corresponding polynomial fit model
          - current fitted peak pixel value
        """
        try:
            ref = calibration_config.get().reference
            p = self.calc.p

            fit: FittedSpectrum = self.list_analyzed_spectras[
                self.current_index
            ].fitted_spectrum

            if ref == "left":
                px_points = p.left_px_points
                freq_points = p.left_freq_points
                poly_coeffs = p.freq_left_peak
                current_px = fit.left_peak_center_px
                title = "Calibration Plot - Left Peak"
                xlabel = "Left peak center px"

            elif ref == "right":
                px_points = p.right_px_points
                freq_points = p.right_freq_points
                poly_coeffs = p.freq_right_peak
                current_px = fit.right_peak_center_px
                title = "Calibration Plot - Right Peak"
                xlabel = "Right peak center px"

            elif ref == "distance":
                px_points = p.dist_px_points
                freq_points = p.dist_freq_points
                poly_coeffs = p.freq_peak_distance
                current_px = fit.inter_peak_distance
                title = "Calibration Plot - Peak Distance"
                xlabel = "Inter-peak distance px"

            else:
                QMessageBox.warning(
                    self,
                    "Unknown Reference",
                    f"Unknown calibration reference: {ref}"
                )
                return

            if px_points is None or freq_points is None:
                QMessageBox.warning(
                    self,
                    "No Calibration Points",
                    f"No interpolation/reference points found for reference '{ref}'."
                )
                return

            px_points = np.asarray(px_points, dtype=float)
            freq_points = np.asarray(freq_points, dtype=float)

            valid = np.isfinite(px_points) & np.isfinite(freq_points)
            px_points = px_points[valid]
            freq_points = freq_points[valid]

            if px_points.size == 0:
                QMessageBox.warning(
                    self,
                    "No Valid Calibration Points",
                    f"No valid calibration points found for reference '{ref}'."
                )
                return

            fig = Figure(figsize=(7, 5))
            ax = fig.add_subplot(111)

            # Calibration reference / interpolation points
            ax.plot(
                px_points,
                freq_points,
                "ko",
                label="Calibration reference points"
            )

            # Fitted polynomial model
            if poly_coeffs is not None and np.all(np.isfinite(poly_coeffs)):
                x_min = float(np.min(px_points))
                x_max = float(np.max(px_points))

                if current_px is not None and np.isfinite(current_px):
                    x_min = min(x_min, float(current_px))
                    x_max = max(x_max, float(current_px))

                x_fit = np.linspace(x_min, x_max, 300)
                y_fit = np.polyval(poly_coeffs, x_fit)

                ax.plot(
                    x_fit,
                    y_fit,
                    "r--",
                    label=f"Polynomial fit, degree={len(poly_coeffs) - 1}"
                )

            # Current fitted peak pixel value
            if fit.is_success and current_px is not None and np.isfinite(current_px):
                if ref == "left":
                    current_freq = self.calc.compute_freq_shift(fit, "left", self.fitting_mode)
                elif ref == "right":
                    current_freq = self.calc.compute_freq_shift(fit, "right", self.fitting_mode)
                else:
                    current_freq = self.calc.compute_freq_shift(fit, "distance", self.fitting_mode)

                if current_freq is not None and np.isfinite(current_freq):
                    ax.axvline(
                        current_px,
                        color="b",
                        linestyle=":",
                        label=f"Current fitted px = {current_px:.2f}"
                    )
                    ax.plot(
                        current_px,
                        current_freq,
                        "bo",
                        markersize=9,
                        label=f"Current freq = {current_freq:.4f} GHz"
                    )

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frequency shift / microwave freq (GHz)")
            ax.grid(True)
            ax.legend()

            cal_window = QWidget()
            cal_window.setWindowTitle(title)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, cal_window)

            layout = QVBoxLayout(cal_window)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            cal_window.setLayout(layout)
            cal_window.resize(750, 550)
            cal_window.show()

            self.open_hist_windows.append(cal_window)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to show calibration plot:\n{e}"
            )