"""Entry window of the data analyzer.

Loads axial-scan files AND standalone calibration files into one list, and
shows either the sample data or the calibration of whatever is selected:

* "Sample data" on a scan      -> AxialScanViewer
* "Sample data" on a cal file  -> warning (a calibration holds no samples)
* "Calibration" on a cal file  -> CalibrationViewer (re-fitted from its raw
                                  frames with the current configs)
* "Calibration" on a scan      -> CalibrationViewer of the SCAN's own
                                  calibration (re-fitted from the frames it
                                  carries when possible, else the stored
                                  polynomial)

All log output — logger and print() alike — is mirrored into the panel on
the right side of the window.
"""
from __future__ import annotations

import pickle
import sys
import traceback
from dataclasses import dataclass

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QSplitter,
    QVBoxLayout, QWidget,
)

from brillouin_system.calibration.calibration import (
    CalibrationCalculator, CalibrationData, calibrate,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.calibration.config.calibration_config_gui import CalibrationConfigDialog
from brillouin_system.guis.data_analyzer.excel_export_axial_scan import (
    BrillouinExport, export_to_excel, get_excel_row_data,
)
from brillouin_system.guis.data_analyzer.log_panel import LogPanel, install_analyzer_logging
from brillouin_system.guis.data_analyzer.show_axial_scan import AxialScanViewer
from brillouin_system.guis.data_analyzer.show_calibration import CalibrationViewer
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.calibration.calibration import calibration_calculator_for_scan
from brillouin_system.analysis.fit_axial_scan import fit_axial_scan
from brillouin_system.my_dataclasses.axial_scan import AxialScan
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    dict_to_dataclass_tree, load_dict_from_hdf5,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config_gui import (
    FindPeaksConfigDialog,
)

log = get_logger(__name__)


@dataclass
class LoadedEntry:
    kind: str            # "scan" | "calibration"
    obj: object          # AxialScan | CalibrationData
    source_path: str

    @property
    def label(self) -> str:
        if self.kind == "scan":
            scan: AxialScan = self.obj
            return (f"[scan] {getattr(scan, 'i', '?')} - "
                    f"{getattr(scan, 'id', 'no-id')}")
        name = self.source_path.replace("\\", "/").split("/")[-1]
        return f"[cal]  {name}"


class AxialScanManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Axial Scan Manager")
        self.setMinimumSize(1000, 550)

        self.entries: dict[int, LoadedEntry] = {}
        self.next_index: int = 0
        self.open_viewers: dict = {}

        self.init_ui()

    def init_ui(self):
        outer = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # --- left: list + controls ---
        left = QWidget()
        layout = QVBoxLayout(left)

        layout.addWidget(QLabel("Loaded files (axial scans and calibrations):"))
        self.scan_list = QListWidget()
        self.scan_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.scan_list.itemDoubleClicked.connect(lambda _: self.show_selected())
        layout.addWidget(self.scan_list)

        # --- view mode ---
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Show:"))
        self.sample_radio = QRadioButton("Sample data")
        self.sample_radio.setChecked(True)
        self.cal_radio = QRadioButton("Calibration")
        mode_row.addWidget(self.sample_radio)
        mode_row.addWidget(self.cal_radio)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # --- buttons ---
        btn_row = QHBoxLayout()

        self.load_btn = QPushButton("Load File(s)")
        self.load_btn.clicked.connect(self.load_files)
        btn_row.addWidget(self.load_btn)

        self.show_btn = QPushButton("Show Selected")
        self.show_btn.clicked.connect(self.show_selected)
        btn_row.addWidget(self.show_btn)

        self.save_all_btn = QPushButton("Save All to Excel")
        self.save_all_btn.clicked.connect(self.save_all_to_excel)
        btn_row.addWidget(self.save_all_btn)

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_selected)
        btn_row.addWidget(self.remove_btn)

        layout.addLayout(btn_row)

        config_row = QHBoxLayout()

        self.config_btn = QPushButton("Open Calibration Config")
        self.config_btn.clicked.connect(self.open_calibration_config)
        config_row.addWidget(self.config_btn)

        self.fitting_config_btn = QPushButton("Open Fitting Config")
        self.fitting_config_btn.clicked.connect(self.open_fitting_config)
        config_row.addWidget(self.fitting_config_btn)

        layout.addLayout(config_row)

        splitter.addWidget(left)

        # --- right: log panel ---
        splitter.addWidget(LogPanel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([650, 350])

    # --- Loading ---

    @staticmethod
    def _load_file(path: str):
        if path.endswith((".hdf5", ".h5")):
            data_dict = load_dict_from_hdf5(path)
            return dict_to_dataclass_tree(data_dict, known_classes)
        with open(path, "rb") as f:
            return pickle.load(f)

    def _add_entry(self, kind: str, obj, path: str):
        entry = LoadedEntry(kind=kind, obj=obj, source_path=path)
        idx = self.next_index
        self.entries[idx] = entry
        self.next_index += 1

        item = QListWidgetItem(f"{idx} - {entry.label}")
        item.setData(Qt.UserRole, idx)
        item.setToolTip(path)
        self.scan_list.addItem(item)
        log.info(f"Loaded {entry.label} from {path}")

    def load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Axial Scans / Calibrations",
            filter="Data Files (*.pkl *.hdf5 *.h5);;All Files (*)")
        if not paths:
            return

        for path in paths:
            try:
                loaded = self._load_file(path)
                objs = loaded if isinstance(loaded, list) else [loaded]

                for obj in objs:
                    if isinstance(obj, AxialScan):
                        self._add_entry("scan", obj, path)
                    elif isinstance(obj, CalibrationData):
                        self._add_entry("calibration", obj, path)
                    else:
                        QMessageBox.warning(
                            self, "Invalid File",
                            f"{path} contained a {type(obj).__name__} — "
                            f"expected AxialScan or CalibrationData.")
            except Exception as e:
                traceback.print_exc()
                QMessageBox.warning(self, "Load Error",
                                    f"Failed to load {path}:\n{e}")

    # --- Showing ---

    def _selected_single_entry(self) -> tuple[int, LoadedEntry] | None:
        items = self.scan_list.selectedItems()
        if not items:
            QMessageBox.information(self, "No Selection",
                                    "Select an entry first.")
            return None
        if len(items) > 1:
            QMessageBox.warning(self, "Multiple Selection",
                                "Please select only one entry to show.")
            return None
        idx = items[0].data(Qt.UserRole)
        entry = self.entries.get(idx)
        if entry is None:
            return None
        return idx, entry

    def show_selected(self):
        selected = self._selected_single_entry()
        if selected is None:
            return
        idx, entry = selected
        show_calibration = self.cal_radio.isChecked()

        try:
            if entry.kind == "scan":
                if show_calibration:
                    self._show_scan_calibration(idx, entry.obj)
                else:
                    self._show_scan(idx, entry.obj)
            else:  # calibration file
                if show_calibration:
                    self._show_calibration_file(idx, entry)
                else:
                    msg = (f"'{entry.label}' is a calibration file — it holds "
                           f"no sample measurements. Switch to 'Calibration' "
                           f"to view it.")
                    log.warning(msg)
                    QMessageBox.warning(self, "No Sample Measurements", msg)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self, "Cannot Open",
                f"Failed to open entry {idx}:\n\n{type(e).__name__}: {e}")

    def _register_window(self, key, window):
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(lambda _, k=key: self.open_viewers.pop(k, None))
        window.show()
        window.raise_()
        self.open_viewers[key] = window

    def _raise_if_open(self, key) -> bool:
        if key in self.open_viewers:
            try:
                self.open_viewers[key].raise_()
                self.open_viewers[key].activateWindow()
                log.info(f"Window {key} is already open — raising it.")
                return True
            except RuntimeError:
                self.open_viewers.pop(key, None)
        return False

    def _show_scan(self, idx: int, scan: AxialScan):
        key = ("scan", idx)
        if self._raise_if_open(key):
            return
        # The viewer fits the whole scan in its constructor, so a bad fitting
        # config (e.g. a pixel-response sample model against a lorentzian
        # reference) raises here — the caller reports it instead of letting
        # the exception escape the slot (PyQt aborts the process on that).
        viewer = AxialScanViewer(scan)
        self._register_window(key, viewer)

    def _show_scan_calibration(self, idx: int, scan: AxialScan):
        key = ("scan-cal", idx)
        if self._raise_if_open(key):
            return
        if scan.calibration_data is None and scan.calibration_params is None:
            QMessageBox.warning(self, "No Calibration",
                                f"Scan '{scan.id}' carries no calibration data "
                                f"and no stored calibration parameters.")
            return
        log.info(f"Showing the calibration of scan '{scan.id}'")
        fitter = SpectrumFitter()
        calc = calibration_calculator_for_scan(
            scan.calibration_data, scan.calibration_params, fitter)
        viewer = CalibrationViewer(
            calc, title=f"Calibration of Scan {scan.id}",
            calibration_data=scan.calibration_data, fitter=fitter)
        self._register_window(key, viewer)

    def _show_calibration_file(self, idx: int, entry: LoadedEntry):
        key = ("cal", idx)
        if self._raise_if_open(key):
            return
        degree = calibration_config.get().degree
        log.info(f"Re-fitting {entry.label} with the current configs "
                 f"(degree={degree})")
        fitter = SpectrumFitter()
        calc = CalibrationCalculator(
            calibrate(data=entry.obj, polyfit_degree=degree, fitter=fitter))
        viewer = CalibrationViewer(calc, title=f"Calibration - {entry.label}",
                                   calibration_data=entry.obj, fitter=fitter)
        self._register_window(key, viewer)

    # --- List management ---

    def remove_selected(self):
        for item in self.scan_list.selectedItems():
            idx = item.data(Qt.UserRole)
            self.entries.pop(idx, None)
            self.scan_list.takeItem(self.scan_list.row(item))

    # --- Configs ---

    def open_calibration_config(self):
        def on_apply(_):
            log.info("[AxialScanManager] Apply has no effect — save the "
                     "configs so they affect the viewer.")
        dlg = CalibrationConfigDialog(on_apply=on_apply, parent=self)
        dlg.exec_()

    def open_fitting_config(self):
        def on_apply(_):
            log.info("[AxialScanManager] Apply has no effect — save the "
                     "configs so they affect the viewer.")
        dlg = FindPeaksConfigDialog(on_apply=on_apply, parent=self)
        dlg.exec_()

    # --- Excel export ---

    def _build_export_rows_for_scan(self, scan: AxialScan) -> list[BrillouinExport]:
        analyzed_spectra = fit_axial_scan(scan)
        return [
            get_excel_row_data(axial_scan=scan, analyzed_spectrum=analyzed, idx=i)
            for i, analyzed in enumerate(analyzed_spectra)
        ]

    def save_all_to_excel(self):
        try:
            scan_entries = {i: e for i, e in self.entries.items()
                            if e.kind == "scan"}
            n_cal = len(self.entries) - len(scan_entries)
            if n_cal:
                log.info(f"Skipping {n_cal} calibration file(s) — Excel export "
                         f"covers sample scans only.")
            if not scan_entries:
                QMessageBox.information(self, "No Scans",
                                        "There are no loaded scans to export.")
                return

            all_rows: list[BrillouinExport] = []
            for idx in sorted(scan_entries.keys()):
                all_rows.extend(
                    self._build_export_rows_for_scan(scan_entries[idx].obj))
            if not all_rows:
                QMessageBox.warning(self, "No Data",
                                    "There is no data to export.")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save All to Excel",
                "all_axial_scans_brillouin_export.xlsx",
                "Excel Files (*.xlsx)")
            if not file_path:
                return
            if not file_path.lower().endswith(".xlsx"):
                file_path += ".xlsx"

            export_to_excel(all_rows, file_path)
            QMessageBox.information(
                self, "Excel Saved",
                f"Saved {len(all_rows)} rows from {len(scan_entries)} "
                f"scan(s) to:\n{file_path}")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Save Failed",
                                 f"Could not save Excel file:\n{e}")


def install_exception_hook():
    """Show unhandled exceptions instead of letting PyQt kill the process.

    PyQt5 calls qFatal() when an exception escapes a slot, which ends the app
    with exit code 0xC0000409 and no visible message.
    """
    def hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        QMessageBox.critical(
            None, "Unhandled Error", f"{exc_type.__name__}: {exc}"
        )

    sys.excepthook = hook


if __name__ == "__main__":
    install_analyzer_logging()
    install_exception_hook()
    app = QApplication(sys.argv)
    window = AxialScanManager()
    window.show()
    sys.exit(app.exec_())
