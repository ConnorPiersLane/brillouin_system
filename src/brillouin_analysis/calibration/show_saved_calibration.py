import sys
import pickle
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from brillouin_system.calibration.calibration import (
    get_calibration_calculator_from_data,
    CalibrationData,
    MeasurementsPerFreq,
    CalibrationMeasurementPoint,
)
from brillouin_system.calibration.calibration_plotting import render_calibration_to_pixmap, CalibrationImageDialog
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    load_dict_from_hdf5,
    dict_to_dataclass_tree,
)


def load_calibration_file(file_path: str) -> CalibrationData:
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    if file_path.endswith((".h5", ".hdf5")):
        native = load_dict_from_hdf5(file_path)


        return dict_to_dataclass_tree(native, known=known_classes)

    raise ValueError(f"Unsupported file type: {file_path}")


def main():
    app = QApplication(sys.argv)

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Open Calibration File",
        "",
        "Calibration Files (*.pkl *.h5 *.hdf5);;All Files (*)"
    )

    if not file_path:
        return

    try:
        calibration_data = load_calibration_file(file_path)
        calib_config = calibration_config.get()
        reference = calib_config.reference
        degree = calib_config.degree
        mode = calib_config.mode

        calculator = get_calibration_calculator_from_data(calibration_data, degree)


        pixmap = render_calibration_to_pixmap(
            calibration_data,
            calculator,
            reference,
            mode=mode,
        )

        dlg = CalibrationImageDialog(pixmap)
        dlg.exec_()

    except Exception as e:
        QMessageBox.critical(None, "Error", str(e))

    sys.exit(0)


if __name__ == "__main__":
    main()