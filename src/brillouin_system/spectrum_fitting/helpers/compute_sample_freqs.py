
from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum


def compute_freq_shift(fitting: FittedSpectrum, calibration_calculator: CalibrationCalculator) -> float | None:
    if not fitting.is_success or calibration_calculator is None:
        return None

    config = calibration_config.get()

    if config.reference == "left":
        return calibration_calculator.freq_left_peak(fitting.left_peak_center_px)
    elif config.reference == "right":
        return calibration_calculator.freq_right_peak(fitting.right_peak_center_px)
    elif config.reference == "distance":
        return calibration_calculator.freq_peak_distance(fitting.inter_peak_distance)
    else:
        return None
