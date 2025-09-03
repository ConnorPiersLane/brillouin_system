# analyzer_manager.py

import pickle
from PyQt5.QtWidgets import QFileDialog

from brillouin_system.my_dataclasses.human_interface_measurements import AxialScan, AnalyzedAxialScan
from brillouin_system.calibration.calibration import CalibrationData, CalibrationCalculator, \
    get_calibration_calculator_from_data
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree

from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes


class AnalyzerManager:
    def __init__(self):
        self.calibration_data_from_file: CalibrationData | None = None
        self.calibration_calculator_from_file: CalibrationCalculator | None = None
        self.stored_axial_scans = []
        self.series_filenames: dict[int, str] = {}  # Maps series index to filename

        self.analyzed_series_lookup: dict[int, AnalyzedAxialScan] = {}

    def load_measurement_series(self, axial_scan: AxialScan):
        self.stored_axial_scans.append(axial_scan)

    def remove_measurement_series(self, index: int):
        if 0 <= index < len(self.stored_axial_scans):
            del self.stored_axial_scans[index]

    def calibrate_from_file(self, calibration: CalibrationData):
        self.calibration_calculator_from_file = get_calibration_calculator_from_data(calibration)

    def get_current_calibration(self, use_series: bool, selected_index: int):
        if use_series:
            if 0 <= selected_index < len(self.stored_axial_scans):
                return self.stored_axial_scans[selected_index].calibration_params
            else:
                raise ValueError(f"index out of range: {selected_index} from {len(self.stored_axial_scans)}")
        return self.calibration_data_from_file

    def load_calibration_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Load Calibration", filter="Supported Files (*.pkl *.hdf5 *.h5);;All Files (*)"
        )
        if not path:
            return None
        try:
            if path.endswith((".hdf5", ".h5")):
                data_dict = load_dict_from_hdf5(path)
                calibration = dict_to_dataclass_tree(data_dict, known_classes)
            else:
                with open(path, "rb") as f:
                    calibration = pickle.load(f)

            self.calibration_data_from_file = calibration
            print(f"[\u2713] Loaded calibration from {path}")
            return path
        except Exception as e:
            print(f"[Analyzer Manager] Failed to load calibration: {e}")
            return None

    def displayed_series_info(self, axial_scan: AxialScan, file_name: str = "Unknown") -> str:
        id = axial_scan.id
        power = axial_scan.power_mW
        expo = axial_scan.system_state.andor_camera_info.exposure
        n = len(axial_scan.measurements)
        return f"File: {file_name} - ID: {id} - Expo: {round(expo, ndigits=3)}[s] - Power: {power}[mW] - N: {n}"

    def load_measurements_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Load Measurement Series", filter="Supported Files (*.pkl *.hdf5 *.h5);;All Files (*)"
        )
        if not path:
            return []

        info_strings = []
        try:
            if path.endswith((".hdf5", ".h5")):
                data_dict = load_dict_from_hdf5(path)
                loaded = dict_to_dataclass_tree(data_dict, known_classes)
            else:
                with open(path, "rb") as f:
                    loaded = pickle.load(f)

            if isinstance(loaded, list):
                self.stored_axial_scans.extend(loaded)
                filename = path.split("/")[-1]
                for series in loaded:
                    self.stored_axial_scans.append(series)
                    series_index = len(self.stored_axial_scans) - 1
                    self.series_filenames[series_index] = filename

                    info_str = self.displayed_series_info(series, file_name=filename)
                    info_strings.append(info_str)
                    print(f"[\u2713] Loaded: {info_str}")

        except Exception as e:
            print(f"[Analyzer Manager] Failed to load measurement series: {e}")
        return info_strings


    def analyze_axial_scan(self,
                           axial_scan: AxialScan,
                           is_do_bg_subtraction: bool,
                           is_use_own_calibration_data: bool,
                           ) -> AnalyzedAxialScan | None:

        if is_use_own_calibration_data:
            if axial_scan.calibration_params is None:
                print("No Calibration Data available in this measurement series")
                return None
            else:
                calibration_calculator = get_calibration_calculator_from_data(axial_scan.calibration_params)
        else:
            if self.calibration_calculator_from_file is None:
                print("No Calibration available from file. Load File and Calibrate")
                return None
            else:
                calibration_calculator = self.calibration_calculator_from_file

        return analyze_axial_scan(scan=axial_scan,
                                  calibration_calculator=calibration_calculator,
                                  do_bg_subtraction=is_do_bg_subtraction)

    def analyze_selected_series(
        self,
        indices: list[int],
        is_do_bg_subtraction: bool,
        is_use_own_calibration_data: bool
    ):
        """
        Analyze and cache spectrum fit results for the given series indices.

        Parameters
        ----------
        indices : list[int]
            Indices of series in stored_measurement_series to analyze.
        is_do_bg_subtraction : bool
            Whether to subtract the background image.
        is_use_own_calibration_data : bool
            Whether to use the series' internal calibration data.
        """
        for index in indices:

            if not (0 <= index < len(self.stored_axial_scans)):
                print(f"[AnalyzerManager] Invalid index {index}")
                continue

            scan = self.stored_axial_scans[index]
            analyzed_scan = self.analyze_axial_scan(
                axial_scan=scan,
                is_do_bg_subtraction=is_do_bg_subtraction,
                is_use_own_calibration_data=is_use_own_calibration_data
            )
            self.analyzed_series_lookup[scan.i] = analyzed_scan