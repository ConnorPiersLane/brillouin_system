import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QListWidget, QRadioButton, QButtonGroup, QMessageBox, QCheckBox, QComboBox, QListView, QDialog
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from brillouin_system.gui.data_analyzer.analyzer_manager import AnalyzerManager

from brillouin_system.gui.helpers.gui_utils import numpy_array_to_pixmap

from brillouin_system.calibration.calibration import (
    CalibrationCalculator, render_calibration_to_pixmap,
    CalibrationImageDialog, calibrate,
)
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.my_dataclasses.human_interface_measurements import AnalyzedAxialScan, AnalyzedMeasurementPoint, \
    MeasurementPoint
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config_gui import FindPeaksConfigDialog


class PlotWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Series Plot")
        self.setMinimumSize(640, 480)

        layout = QVBoxLayout(self)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def closeEvent(self, event):
        # Notify parent that the window is gone
        if self.parent():
            self.parent().plot_window = None
        event.accept()

class HistogramWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histogram Plot")
        self.setMinimumSize(640, 480)

        layout = QVBoxLayout(self)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def closeEvent(self, event):
        # Notify parent that the window is gone
        if self.parent():
            self.parent().histogram_window = None
        event.accept()



class AnalyzerViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyzer Viewer")
        self.setMinimumSize(1200, 720)
        self.analyze_manager = AnalyzerManager()

        self.current_point_index = 0
        self.selected_analyzed_scan: AnalyzedAxialScan | None = None
        self.current_series = None
        self.analyzed_series_lookup = {}

        self.plot_window = None
        self.histogram_window = None

        self.init_ui()



    def init_ui(self):
        self.setup_main_layout()
        self.setup_display_area()
        self.setup_series_controls()
        self.setup_calibration_controls()
        self.setup_single_series_controls()
        self.setup_connections()

    def setup_main_layout(self):
        self.outer_layout = QVBoxLayout()
        self.setLayout(self.outer_layout)

        self.display_row = QHBoxLayout()
        self.outer_layout.addLayout(self.display_row)

        self.bottom_row = QHBoxLayout()
        self.outer_layout.addLayout(self.bottom_row)

    def setup_display_area(self):
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


        self.display_row.addLayout(left_display_col)
        self.display_row.addWidget(self.mako_label)

    def setup_series_controls(self):
        self.series_group = QGroupBox("Series Controls")
        series_layout = QVBoxLayout()

        self.series_list_widget = QListWidget()
        self.series_list_widget.setSelectionMode(QListWidget.MultiSelection)

        self.add_series_btn = QPushButton("Add Series")
        self.remove_series_btn = QPushButton("Remove Selected")

        series_layout.addWidget(self.series_list_widget)
        series_layout.addWidget(self.add_series_btn)
        series_layout.addWidget(self.remove_series_btn)
        self.series_group.setLayout(series_layout)

        self.bottom_row.addWidget(self.series_group)

    def setup_calibration_controls(self):
        self.calibration_group = QGroupBox("Calibration and Fitting")
        calibration_layout = QVBoxLayout()

        self.radio_use_calibration_in_series = QRadioButton("Use Series Calibration")
        self.radio_use_external = QRadioButton("Use External Calibration")
        self.radio_use_calibration_in_series.setChecked(True)

        self.calibration_button_group = QButtonGroup()
        self.calibration_button_group.addButton(self.radio_use_calibration_in_series)
        self.calibration_button_group.addButton(self.radio_use_external)

        self.load_calibration_btn = QPushButton("Load Calibration")
        self.loaded_calibration_label = QLabel("Loaded Calibration: None")
        self.show_calibration_btn = QPushButton("Show Calibration")
        self.config_series_btn = QPushButton("Config")
        self.bg_sub_checkbox = QCheckBox("Do BG Subtraction")
        self.bg_sub_checkbox.setChecked(False)
        self.bg_sub_checkbox.setToolTip("Enable background subtraction before spectrum fitting.")
        self.analyze_series_btn = QPushButton("Analyze Selected Series")
        self.clear_analysis_btn = QPushButton("Clear Analyzed")

        calibration_layout.addWidget(self.radio_use_calibration_in_series)
        calibration_layout.addWidget(self.radio_use_external)
        calibration_layout.addWidget(self.load_calibration_btn)
        calibration_layout.addWidget(self.loaded_calibration_label)
        calibration_layout.addWidget(self.show_calibration_btn)
        calibration_layout.addWidget(self.config_series_btn)
        calibration_layout.addWidget(self.bg_sub_checkbox)
        calibration_layout.addWidget(self.analyze_series_btn)
        calibration_layout.addWidget(self.clear_analysis_btn)

        self.calibration_group.setLayout(calibration_layout)
        self.bottom_row.addWidget(self.calibration_group)

    def setup_single_series_controls(self):
        self.single_series_group = QGroupBox("One Measurement Series")
        layout = QVBoxLayout()

        self.series_status_label = QLabel("Show Series: No series selected.")
        layout.addWidget(self.series_status_label)

        self.analyzed_series_dropdown = QComboBox()
        self.analyzed_series_dropdown.setFixedWidth(250)
        self.analyzed_series_dropdown.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.analyzed_series_dropdown.setView(QListView())
        self.analyzed_series_dropdown.view().setMinimumWidth(600)

        layout.addWidget(QLabel("Analyzed Series:"))
        layout.addWidget(self.analyzed_series_dropdown)

        self.show_selected_btn = QPushButton("Show")
        layout.addWidget(self.show_selected_btn)

        nav_layout = QHBoxLayout()
        self.left_btn = QPushButton("Left")
        self.right_btn = QPushButton("Right")
        self.index_label = QLabel("0 of 0")
        nav_layout.addWidget(self.left_btn)
        nav_layout.addWidget(self.right_btn)
        nav_layout.addWidget(self.index_label)
        layout.addLayout(nav_layout)

        # X/Y axis selection
        xy_layout = QHBoxLayout()
        self.x_axis_label = QLabel("X:")
        self.x_axis_dropdown = QComboBox()
        self.x_axis_dropdown.addItem("index")
        self.y_axis_label = QLabel("Y:")
        self.y_axis_dropdown = QComboBox()

        y_axis_options = [
            "freq_shift_left_peak_ghz",
            "freq_shift_right_peak_ghz",
            "freq_shift_peak_distance_ghz",
            "fwhm_left_peak_ghz",
            "fwhm_right_peak_ghz",
            "left_peak_photons",
            "right_peak_photons",
            "total_photons"
        ]
        for opt in y_axis_options:
            self.y_axis_dropdown.addItem(opt)

        xy_layout.addWidget(self.x_axis_label)
        xy_layout.addWidget(self.x_axis_dropdown)
        xy_layout.addWidget(self.y_axis_label)
        xy_layout.addWidget(self.y_axis_dropdown)

        layout.addLayout(xy_layout)
        self.single_series_group.setLayout(layout)

        self.plot_series_btn = QPushButton("Plot Series")
        layout.addWidget(self.plot_series_btn)

        self.plot_histogram_btn = QPushButton("Plot Histogram")
        layout.addWidget(self.plot_histogram_btn)

        self.show_stats_checkbox = QCheckBox("Show Mean ± Std")
        layout.addWidget(self.show_stats_checkbox)

        self.bottom_row.addWidget(self.single_series_group)

    def setup_connections(self):
        self.add_series_btn.clicked.connect(self.load_series_file)
        self.remove_series_btn.clicked.connect(self.remove_selected_series)
        self.load_calibration_btn.clicked.connect(self.load_calibration_file)
        self.show_calibration_btn.clicked.connect(self.show_calibration)
        self.config_series_btn.clicked.connect(self.open_config_dialog)
        self.analyze_series_btn.clicked.connect(self.analyze_selected_series)
        self.clear_analysis_btn.clicked.connect(self.clear_analyzed_series)
        self.show_selected_btn.clicked.connect(self.show_selected_analyzed_series)
        self.left_btn.clicked.connect(self.go_left)
        self.right_btn.clicked.connect(self.go_right)
        self.plot_series_btn.clicked.connect(self.plot_selected_series)
        self.plot_histogram_btn.clicked.connect(self.plot_histogram)

    def open_config_dialog(self):
        dialog = FindPeaksConfigDialog(self)
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
            series = self.analyze_manager.stored_axial_scans[index]
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

        indices = [self.series_list_widget.row(item) for item in selected_items]

        use_series_calibration = self.radio_use_calibration_in_series.isChecked()
        do_bg_sub = self.bg_sub_checkbox.isChecked()

        self.analyze_manager.analyze_selected_series(
            indices=indices,
            is_do_bg_subtraction=do_bg_sub,
            is_use_own_calibration_data=use_series_calibration
        )

        self.series_status_label.setText(f"Analyzed {len(indices)} series.")
        self.refresh_analyzed_dropdown()

    def refresh_analyzed_dropdown(self):
        self.analyzed_series_dropdown.clear()

        for index in sorted(self.analyze_manager.analyzed_series_lookup):
            # Make sure the index still exists in stored_axial_scans
            if 0 <= index < len(self.analyze_manager.stored_axial_scans):
                series = self.analyze_manager.stored_axial_scans[index]
                filename = self.analyze_manager.series_filenames.get(index, "Unknown")
                label = self.analyze_manager.displayed_series_info(series, file_name=filename)

                self.analyzed_series_dropdown.addItem(label, userData=index)
                self.analyzed_series_dropdown.setItemData(
                    self.analyzed_series_dropdown.count() - 1,
                    label,
                    Qt.ToolTipRole  # Tooltip on hover
                )

    def clear_analyzed_series(self):
        self.analyze_manager.analyzed_series_lookup.clear()
        self.analyzed_series_dropdown.clear()
        self.series_status_label.setText("Cleared analyzed series.")

    def show_selected_analyzed_series(self):
        index = self.analyzed_series_dropdown.currentData()
        if index is None:
            QMessageBox.warning(self, "Selection Error", "No analyzed series selected.")
            return

        self.current_series = self.analyze_manager.stored_axial_scans[index]
        self.selected_analyzed_scan = self.analyze_manager.analyzed_series_lookup[index]
        self.current_point_index = 0
        self.series_status_label.setText("Show Series: OK.")
        self.update_series_display()

    def display_result(self, measurement_point: MeasurementPoint, analyzed_measurement_point: AnalyzedMeasurementPoint):
        frame = measurement_point.frame_andor

        pixmap = numpy_array_to_pixmap(frame)
        self.frame_label.setPixmap(pixmap.scaled(
            self.frame_label.width(), self.frame_label.height(), Qt.KeepAspectRatio
        ))

        x_px = analyzed_measurement_point.fitted_spectrum.x_pixels
        spectrum = analyzed_measurement_point.fitted_spectrum.sline

        self.ax.clear()
        self.ax.plot(x_px, spectrum, 'k.', label="Spectrum")

        interpeak = None
        freq_shift_ghz = None

        if analyzed_measurement_point.fitted_spectrum.is_success:
            x_fit_refined = analyzed_measurement_point.fitted_spectrum.x_fit_refined
            y_fit_refined = analyzed_measurement_point.fitted_spectrum.y_fit_refined
            interpeak = analyzed_measurement_point.fitted_spectrum.inter_peak_distance
            freq_shift_ghz = analyzed_measurement_point.freq_shifts.freq_shift_peak_distance_ghz
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
        if self.current_point_index > 0:
            self.current_point_index -= 1
            self.update_series_display()

    def go_right(self):
        if self.current_series and self.current_point_index < len(self.current_series.measurements) - 1:
            self.current_point_index += 1
            self.update_series_display()

    def update_series_display(self):
        if not self.selected_analyzed_scan:
            self.index_label.setText("0 of 0")
            return
        self.index_label.setText(f"{self.current_point_index + 1} of {len(self.selected_analyzed_scan.analyzed_measurements)}")
        point_to_be_displayed = self.selected_analyzed_scan.axial_scan.measurements[self.current_point_index]
        result_to_be_displayed = self.selected_analyzed_scan.analyzed_measurements[self.current_point_index]
        self.display_result(measurement_point=point_to_be_displayed,
                            analyzed_measurement_point=result_to_be_displayed)
        print(f"Index: {self.current_point_index}")


    def plot_selected_series(self):
        index = self.analyzed_series_dropdown.currentData()
        if index is None:
            QMessageBox.warning(self, "Selection Error", "No analyzed series selected.")
            return

        self.selected_analyzed_scan = self.analyze_manager.analyzed_series_lookup[index]

        x_axis = self.x_axis_dropdown.currentText()
        y_axis = self.y_axis_dropdown.currentText()

        if x_axis != "index":
            QMessageBox.warning(self, "Unsupported X Axis", "Only 'index' is currently supported for X.")
            return

        # Extract Y values
        y_values = [
            getattr(af, y_axis)
            for af in self.selected_analyzed_scan
            if getattr(af, y_axis) is not None
        ]

        if not y_values:
            QMessageBox.warning(self, "No Data", f"No valid data found for '{y_axis}'.")
            return

        x_values = list(range(len(y_values)))

        # Determine plot category
        if "photon" in y_axis.lower():
            current_category = "photons"
        else:
            current_category = "ghz"

        # Create new plot window if switching categories
        if (
                self.plot_window is None or
                self.last_y_axis_category != current_category
        ):
            if self.plot_window is not None:
                self.plot_window.close()
                self.plot_window = None

            self.plot_window = PlotWindow(self)
            self.plot_window_ax = self.plot_window.ax
            self.plot_window_ax.set_title("Series Plot")
            self.plot_window_ax.set_xlabel(x_axis)
            self.plot_window_ax.set_ylabel("Photons" if current_category == "photons" else "GHz")
            self.plot_window.show()
            self.last_y_axis_category = current_category
        else:
            self.plot_window_ax = self.plot_window.ax

        # Plot the data
        line, = self.plot_window_ax.plot(x_values, y_values, 'o--', label=y_axis)

        # Show stats overlay if checkbox is checked
        if self.show_stats_checkbox.isChecked():
            from brillouin_system.my_dataclasses.analyzer_results import analyze_frame_statistics
            stats = analyze_frame_statistics(self.selected_analyzed_scan)

            # Get corresponding mean and std field dynamically
            mean_attr = f"mean_{y_axis}"
            std_attr = f"std_{y_axis}"

            mean_val = getattr(stats, mean_attr, None)
            std_val = getattr(stats, std_attr, None)

            if mean_val is not None and std_val is not None:
                ymin = mean_val - std_val
                ymax = mean_val + std_val
                color = line.get_color()

                self.plot_window_ax.axhline(mean_val, color=color, linestyle='--', alpha=0.5, label=f"Mean: {round(mean_val,ndigits=3)}")
                self.plot_window_ax.fill_between(x_values, ymin, ymax, color=color, alpha=0.2, label=f"±1 STD: {round(std_val,ndigits=3)}")

        self.plot_window_ax.legend()
        self.plot_window.canvas.draw()

    def plot_histogram(self):
        index = self.analyzed_series_dropdown.currentData()
        if index is None:
            QMessageBox.warning(self, "Selection Error", "No analyzed series selected.")
            return

        self.selected_analyzed_scan = self.analyze_manager.analyzed_series_lookup[index]
        y_axis = self.y_axis_dropdown.currentText()

        y_values = [
            getattr(af, y_axis)
            for af in self.selected_analyzed_scan
            if getattr(af, y_axis) is not None
        ]

        if not y_values:
            QMessageBox.warning(self, "No Data", f"No valid data found for '{y_axis}'.")
            return

        y_array = np.array(y_values)

        # Create the histogram window if it doesn't exist
        if self.histogram_window is None:
            self.histogram_window = HistogramWindow(self)
            self.histogram_ax = self.histogram_window.ax
            self.histogram_ax.set_title("Histogram")
            self.histogram_ax.set_xlabel(y_axis)
            self.histogram_ax.set_ylabel("Frequency")
            self.histogram_window.show()
        else:
            self.histogram_ax = self.histogram_window.ax

        # Plot histogram
        color = np.random.rand(3, )  # generate a random color
        n, bins, patches = self.histogram_ax.hist(y_array, bins=20, alpha=0.5, label=y_axis, color=color,
                                                  edgecolor='black')

        # Plot Gaussian overlay if enabled
        if self.show_stats_checkbox.isChecked():
            from brillouin_system.my_dataclasses.analyzer_results import analyze_frame_statistics
            stats = analyze_frame_statistics(self.selected_analyzed_scan)

            mean_attr = f"mean_{y_axis}"
            std_attr = f"std_{y_axis}"

            mean_val = getattr(stats, mean_attr, None)
            std_val = getattr(stats, std_attr, None)

            if mean_val is not None and std_val is not None:
                from scipy.stats import norm
                x = np.linspace(min(y_array), max(y_array), 200)
                pdf = norm.pdf(x, loc=mean_val, scale=std_val)
                scaled_pdf = pdf * len(y_array) * (bins[1] - bins[0])
                self.histogram_ax.plot(x, scaled_pdf, '--', color=color, label=f"Gaussian ({y_axis})")

        self.histogram_ax.legend()
        self.histogram_window.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = AnalyzerViewer()
    viewer.show()
    sys.exit(app.exec_())
