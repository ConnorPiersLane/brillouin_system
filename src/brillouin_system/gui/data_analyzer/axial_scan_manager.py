import sys
import pickle
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt

from brillouin_system.calibration.config.calibration_config_gui import CalibrationConfigDialog
# Reuse your existing modules

from brillouin_system.gui.helpers.show_axial_scan import AxialScanViewer

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
        dlg = CalibrationConfigDialog(self)
        dlg.exec_()

    def open_fitting_config(self):
        def on_apply(_):
            # No need to update anything locally
            print("[AxialScanManager] Pressed Apply — configs are live globally.")

        dlg = FindPeaksConfigDialog(on_apply=on_apply, parent=self)
        dlg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AxialScanManager()
    window.show()
    sys.exit(app.exec_())
