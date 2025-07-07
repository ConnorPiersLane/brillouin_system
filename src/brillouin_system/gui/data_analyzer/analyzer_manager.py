# analyzer_manager.py

import pickle
from PyQt5.QtWidgets import QFileDialog

from brillouin_system.fitting.fit_util import get_sline_from_image
from brillouin_system.fitting.fitting_manager import fit_reference_spectrum, fit_sample_spectrum, get_empty_fitting
from brillouin_system.my_dataclasses.analyzer_results import AnalyzedFrame, fitting_to_analyzer_result
from brillouin_system.my_dataclasses.measurements import MeasurementSeries
from brillouin_system.my_dataclasses.calibration import CalibrationData, CalibrationCalculator, \
    get_calibration_calculator_from_data
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import load_dict_from_hdf5, dict_to_dataclass_tree

from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes


class AnalyzerManager:
    def __init__(self):
        self.calibration_data_from_file: CalibrationData | None = None
        self.calibration_calculator_from_file: CalibrationCalculator | None = None
        self.stored_measurement_series = []
        self.series_filenames: dict[int, str] = {}  # Maps series index to filename

        self.analyzed_series_lookup: dict[int, list[AnalyzedFrame]] = {}

    def load_measurement_series(self, measurement_series: MeasurementSeries):
        self.stored_measurement_series.append(measurement_series)

    def remove_measurement_series(self, index: int):
        if 0 <= index < len(self.stored_measurement_series):
            del self.stored_measurement_series[index]

    def calibrate_from_file(self, calibration: CalibrationData):
        self.calibration_calculator_from_file = get_calibration_calculator_from_data(calibration)

    def get_current_calibration(self, use_series: bool, selected_index: int):
        if use_series:
            if 0 <= selected_index < len(self.stored_measurement_series):
                return self.stored_measurement_series[selected_index].calibration_data
            else:
                raise ValueError(f"index out of range: {selected_index} from {len(self.stored_measurement_series)}")
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
                filename = path.split("/")[-1]
                for series in loaded:
                    self.stored_measurement_series.append(series)
                    series_index = len(self.stored_measurement_series) - 1
                    self.series_filenames[series_index] = filename

                    info_str = self.displayed_series_info(series, file_name=filename)
                    info_strings.append(info_str)
                    print(f"[\u2713] Loaded: {info_str}")

        except Exception as e:
            print(f"[Analyzer Manager] Failed to load measurement series: {e}")
        return info_strings


    def analyze_frames_in_measurement_series(self,
                                             measurement: MeasurementSeries,
                                             is_do_bg_subtraction: bool,
                                             is_use_own_calibration_data: bool,
                                             ) -> list[AnalyzedFrame]:
        analyzed_frames = []

        if is_use_own_calibration_data:
            if measurement.calibration_data is None:
                print("No Calibration Data available in this measurement series")
                return []
            else:
                calibration_calculator = get_calibration_calculator_from_data(measurement.calibration_data)
        else:
            if self.calibration_calculator_from_file is None:
                print("No Calibration available from file. Load File and Calibrate")
                return []
            else:
                calibration_calculator = self.calibration_calculator_from_file

        for mp in measurement.measurements:
            frame = mp.frame
            if is_do_bg_subtraction:
                bg_image = measurement.state_mode.bg_image.mean_image
                if bg_image is None:
                    print("No BG Image availalbe")
                    return analyzed_frames
                frame = frame - bg_image

            sline = get_sline_from_image(frame)

            try:
                if measurement.state_mode.is_reference_mode:
                    fs = fit_reference_spectrum(sline=sline)
                else:
                    fs = fit_sample_spectrum(sline=sline, calibration_calculator=calibration_calculator)
            except Exception as e:
                print(f"[BrillouinManager] Fitting error: {e}")
                fs = get_empty_fitting(sline)

            af = fitting_to_analyzer_result(frame=frame, fitting=fs, calibration_calculator=calibration_calculator,
                                            camera_settings=measurement.state_mode.camera_settings)
            analyzed_frames.append(af)
        return analyzed_frames





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
            # if index in self.analyzed_series_lookup:
            #     continue  # already analyzed

            if not (0 <= index < len(self.stored_measurement_series)):
                print(f"[AnalyzerManager] Invalid index {index}")
                continue

            series = self.stored_measurement_series[index]
            analyzed_frames = self.analyze_frames_in_measurement_series(
                measurement=series,
                is_do_bg_subtraction=is_do_bg_subtraction,
                is_use_own_calibration_data=is_use_own_calibration_data
            )
            self.analyzed_series_lookup[index] = analyzed_frames