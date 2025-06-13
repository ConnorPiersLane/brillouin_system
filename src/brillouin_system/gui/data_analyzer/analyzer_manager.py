# analyzer_manager.py

import pickle
from PyQt5.QtWidgets import QFileDialog
from brillouin_system.my_dataclasses.measurements import MeasurementSeries
from brillouin_system.my_dataclasses.calibration import CalibrationResults


class AnalyzerManager:
    def __init__(self):
        self.stored_measurement_series = []
        self.external_calibration: CalibrationResults = None

    def load_measurement_series(self, measurement_series: MeasurementSeries):
        self.stored_measurement_series.append(measurement_series)

    def remove_measurement_series(self, index: int):
        if 0 <= index < len(self.stored_measurement_series):
            del self.stored_measurement_series[index]

    def set_calibration(self, calibration: CalibrationResults):
        self.external_calibration = calibration

    def get_current_calibration(self, use_series: bool, selected_index: int):
        if use_series and 0 <= selected_index < len(self.stored_measurement_series):
            return self.stored_measurement_series[selected_index].calibration_data
        return self.external_calibration

    def load_calibration_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Load Calibration", filter="Pickle Files (*.pkl);;All Files (*)"
        )
        if not path:
            return None
        try:
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
        return f"File: {file_name} - Name: {name} - Expo: {round(expo,ndigits=3)}[s] - Power: {power}[mW]"

    def load_measurements_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Load Measurement Series", filter="Pickle Files (*.pkl);;All Files (*)"
        )
        if not path:
            return []
        info_strings = []
        try:
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
