from dataclasses import dataclass

import numpy as np

from brillouin_system.calibration.calibration import CalibrationData
from brillouin_system.my_dataclasses.analyzed_freq_shifts import AnalyzedFreqShifts
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts


@dataclass
class RequestAxialScan:
    id: str
    power_mW: float
    n_measurements: int
    step_size_um: float

@dataclass
class MeasurementPoint:
    frame_andor: np.ndarray  # Original frame, not subtracted
    lens_zaber_position: float
    frame_left_allied: np.ndarray | None = None
    frame_right_allied: np.ndarray | None = None

@dataclass
class EyeLocation:
    index: int = 0

@dataclass
class AxialScan:
    i: int  # internal tracker
    id: str
    power_mW: float
    measurements: list[MeasurementPoint]
    system_state: SystemState
    calibration_data: CalibrationData | None
    eye_location: None | EyeLocation = None

@dataclass
class AnalyzedMeasurementPoint:
    fitted_spectrum: FittedSpectrum
    freq_shifts: AnalyzedFreqShifts
    photons: PhotonsCounts


@dataclass
class AnalyzedAxialScan:
    axial_scan: AxialScan
    analyzed_measurements: list[AnalyzedMeasurementPoint]
