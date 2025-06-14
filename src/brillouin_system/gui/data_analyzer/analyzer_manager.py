# analyzer_manager.py

import pickle
from PyQt5.QtWidgets import QFileDialog
from brillouin_system.my_dataclasses.measurements import MeasurementSeries, MeasurementPoint
from brillouin_system.my_dataclasses.calibration import CalibrationData, calibrate, CalibrationCalculator, \
    get_calibration_calculator_from_data
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree

from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes


class AnalyzerManager:
    def __init__(self):
        self.stored_measurement_series = []
        self.external_calibration: CalibrationData = None

    def load_measurement_series(self, measurement_series: MeasurementSeries):
        self.stored_measurement_series.append(measurement_series)

    def remove_measurement_series(self, index: int):
        if 0 <= index < len(self.stored_measurement_series):
            del self.stored_measurement_series[index]

    def set_calibration(self, calibration: CalibrationData):
        self.external_calibration = calibration

    def get_current_calibration(self, use_series: bool, selected_index: int):
        if use_series and 0 <= selected_index < len(self.stored_measurement_series):
            return self.stored_measurement_series[selected_index].calibration_data
        return self.external_calibration

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

            self.set_calibration(calibration)
            print(f"[\u2713] Loaded calibration from {path}")
            return path
        except Exception as e:
            print(f"[Analyzer Manager] Failed to load calibration: {e}")
            return None

    def displayed_series_info(self, series: MeasurementSeries, file_name: str = "Unknown") -> str:
        name = series.settings.name if series.settings else "Unnamed"
        power = series.settings.power_mW if series.settings else "?"
        expo = series.state_mode.camera_settings.exposure_time_s if series.state_mode and series.state_mode.camera_settings else "?"
        n = series.settings.n_measurements if series.settings and hasattr(series.settings, "n_measurements") else "?"
        return f"File: {file_name} - Name: {name} - Expo: {round(expo, ndigits=3)}[s] - Power: {power}[mW] - N: {n}"

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
                self.stored_measurement_series.extend(loaded)
                for series in loaded:
                    info_str = self.displayed_series_info(series, file_name=path.split("/")[-1])
                    info_strings.append(info_str)
                    print(f"[\u2713] Loaded: {info_str}")
        except Exception as e:
            print(f"[Analyzer Manager] Failed to load measurement series: {e}")
        return info_strings

    def run_spectrum_fit_on_measurement_series(self,
                                               measurement: MeasurementSeries,
                                               is_do_bg_subtraction: bool,
                                               external_calibration_data = None):




        if external_calibration_data is None and measurement.calibration_data is None:
            print(" No fitting possible")
            return

        if external_calibration_data is None:
            calibration_data = measurement.calibration_data
        else:
            calibration_data = external_calibration_data

        calibration_calculator = get_calibration_calculator_from_data(calibration_data)
        for mp in measurement.measurements:
            pass
