import sys
import pickle
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt

from brillouin_system.calibration.config.calibration_config_gui import CalibrationConfigDialog
# Reuse your existing modules

from brillouin_system.guis.data_analyzer.show_axial_scan import AxialScanViewer

from brillouin_system.my_dataclasses.human_interface_measurements import AxialScan
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config_gui import FindPeaksConfigDialog


class AxialScanManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Axial Scan Manager")
        self.setMinimumSize(800, 500)

        self.scans: dict[int, AxialScan] = {}   # internal tracker → scan
        self.next_index: int = 0

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- List of loaded scans ---
        self.scan_list = QListWidget()
        self.scan_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(QLabel("Loaded Axial Scans:"))
        layout.addWidget(self.scan_list)

        # --- Buttons row ---
        btn_row = QHBoxLayout()

        self.load_btn = QPushButton("Load Axial Scan(s)")
        self.load_btn.clicked.connect(self.load_scans)
        btn_row.addWidget(self.load_btn)

        self.show_btn = QPushButton("Show Axial Scan")
        self.show_btn.clicked.connect(self.show_scan)
        btn_row.addWidget(self.show_btn)


        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_selected)
        btn_row.addWidget(self.remove_btn)

        layout.addLayout(btn_row)

        # --- Config Buttons Row ---
        config_row = QHBoxLayout()

        self.config_btn = QPushButton("Open Calibration Config")
        self.config_btn.clicked.connect(self.open_calibration_config)
        config_row.addWidget(self.config_btn)

        self.fitting_config_btn = QPushButton("Open Fitting Config")
        self.fitting_config_btn.clicked.connect(self.open_fitting_config)
        config_row.addWidget(self.fitting_config_btn)

        layout.addLayout(config_row)

    # --- Logic ---

    def load_scans(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Axial Scans",
            filter="Axial Scan Files (*.pkl *.hdf5 *.h5);;All Files (*)"
        )
        if not paths:
            return

        for path in paths:
            try:
                # ✅ Support HDF5 and Pickle
                if path.endswith((".hdf5", ".h5")):
                    data_dict = load_dict_from_hdf5(path)
                    loaded = dict_to_dataclass_tree(data_dict, known_classes)
                else:
                    with open(path, "rb") as f:
                        loaded = pickle.load(f)

                scans = loaded if isinstance(loaded, list) else [loaded]

                for scan in scans:
                    if not isinstance(scan, AxialScan):
                        QMessageBox.warning(self, "Invalid File", f"{path} contained a non-AxialScan object.")
                        continue

                    idx = self.next_index
                    self.scans[idx] = scan
                    self.next_index += 1

                    label = f"{idx} - {getattr(scan, 'i', '?')} - {getattr(scan, 'id', 'no-id')}"
                    item = QListWidgetItem(label)
                    item.setData(Qt.UserRole, idx)
                    self.scan_list.addItem(item)

            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load {path}:\n{e}")

        import math
        import pandas as pd

        rows = []

        for idx, scan in self.scans.items():  # FIX: items() gives (key, value)
            try:
                lp = scan.eye_tracker_results.laser_position

                # Compute values
                radius = math.sqrt(lp[0] ** 2 + lp[1] ** 2)
                angle = math.degrees(math.atan2(lp[1], lp[0]))

                # ID
                scan_id = f"{getattr(scan, 'i', '?')}-{getattr(scan, 'id', 'no-id')}"

                # is_found (adjust if needed)
                is_found = scan.reflection_result_forwards is not None

                # DAQ values (handle list or scalar)
                daq = scan.reflection_result_forwards.peak_value if is_found else None
                threshold_high = scan.reflection_result_forwards.threshold_high
                threshold_low = scan.reflection_result_forwards.threshold_low
                if isinstance(daq, (list, tuple)):
                    max_daq = max(daq) if daq else None
                else:
                    max_daq = daq

                # Store row
                rows.append({
                    "ID": scan_id,
                    "Angle": angle,
                    "Radius": radius,
                    "is_found": is_found,
                    "max daq signal [V]": max_daq,
                    'threshold_high': threshold_high,
                    "threshold_low": threshold_low,
                })

            except Exception as e:
                print(f"Skipping scan {idx}: {e}")

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Save to Excel
        output_path = "scan_analysis.xlsx"
        df.to_excel(output_path, index=False)

        print(f"Saved Excel to: {output_path}")

    def show_scan(self):
        items = self.scan_list.selectedItems()
        if not items:
            QMessageBox.information(self, "No Selection", "Select one or more scans first.")
            return
        if len(items) > 1:
            QMessageBox.warning(self, "Multiple Selection", "Please select only one scan at a time.")
            return

        if not hasattr(self, "open_viewers"):
            self.open_viewers = {}

        for item in items:
            idx = item.data(Qt.UserRole)
            scan = self.scans.get(idx)
            if not scan:
                continue

            if idx in self.open_viewers:
                try:
                    self.open_viewers[idx].raise_()
                    self.open_viewers[idx].activateWindow()
                    print(f"⚠️ Scan {idx} is already open. Not opening a second viewer.")
                except RuntimeError:
                    print(f"⚠️ Viewer for scan {idx} was closed unexpectedly. Reopening...")
                    viewer = AxialScanViewer(scan)
                    viewer.setAttribute(Qt.WA_DeleteOnClose, True)
                    viewer.destroyed.connect(lambda _, i=idx: self.open_viewers.pop(i, None))
                    viewer.show()
                    viewer.raise_()
                    self.open_viewers[idx] = viewer
            else:
                viewer = AxialScanViewer(scan)
                viewer.setAttribute(Qt.WA_DeleteOnClose, True)
                viewer.destroyed.connect(lambda _, i=idx: self.open_viewers.pop(i, None))
                viewer.show()
                viewer.raise_()
                self.open_viewers[idx] = viewer

    def remove_selected(self):
        selected_items = self.scan_list.selectedItems()
        for item in selected_items:
            idx = item.data(Qt.UserRole)
            if idx in self.scans:
                del self.scans[idx]
            row = self.scan_list.row(item)
            self.scan_list.takeItem(row)

    def open_calibration_config(self):
        def on_apply(_):
            # No need to update anything locally
            print("[AxialScanManager] Apply has no effect — you need to save the configs to they effect the viewer.")
        dlg = CalibrationConfigDialog(on_apply=on_apply, parent=self)
        dlg.exec_()




    def open_fitting_config(self):
        def on_apply(_):
            # No need to update anything locally
            print("[AxialScanManager] Apply has no effect — you need to save the configs to they effect the viewer")

        dlg = FindPeaksConfigDialog(on_apply=on_apply, parent=self)
        dlg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AxialScanManager()
    window.show()
    sys.exit(app.exec_())
