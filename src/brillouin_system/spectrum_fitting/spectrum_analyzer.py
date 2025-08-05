
from brillouin_system.my_dataclasses.analyzed_freq_shifts import AnalyzedFreqShifts

from brillouin_system.calibration.calibration import CalibrationCalculator
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum


class SpectrumAnalyzer:

    def __init__(self,
                 calibration_calculator: CalibrationCalculator,
                 ):
        self.calibration_calculator: CalibrationCalculator = calibration_calculator

    def analyze_spectrum(self,
                      fitting: FittedSpectrum) -> AnalyzedFreqShifts:
        if fitting.is_success:
            return AnalyzedFreqShifts(
                    freq_shift_left_peak_ghz = float(self.calibration_calculator.freq_left_peak(fitting.left_peak_center_px)),
                    freq_shift_right_peak_ghz = float(self.calibration_calculator.freq_right_peak(fitting.right_peak_center_px)),
                    freq_shift_peak_distance_ghz = float(self.calibration_calculator.freq_peak_distance(fitting.inter_peak_distance)),
                    fwhm_left_peak_ghz = float(
                        abs(self.calibration_calculator.df_left_peak(fitting.left_peak_center_px, fitting.left_peak_width_px))
                    ),
                    fwhm_right_peak_ghz = float(
                        abs(self.calibration_calculator.df_right_peak(fitting.right_peak_center_px, fitting.right_peak_width_px))
                    ),
            )
        else:
            return AnalyzedFreqShifts(
                freq_shift_left_peak_ghz=None,
                freq_shift_right_peak_ghz=None,
                freq_shift_peak_distance_ghz=None,
                fwhm_left_peak_ghz=None,
                fwhm_right_peak_ghz=None,
            )
