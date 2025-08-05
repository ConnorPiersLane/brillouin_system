from dataclasses import dataclass

import numpy as np

from brillouin_system.calibration.calibration import CalibrationData, CalibrationCalculator
from brillouin_system.my_dataclasses.analyzed_freq_shifts import AnalyzedFreqShifts
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import PhotonsCounts, \
    calculate_photon_counts_from_fitted_spectrum
from brillouin_system.spectrum_fitting.helpers.subtract_background import subtract_background
from brillouin_system.spectrum_fitting.spectrum_analyzer import SpectrumAnalyzer
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


# -------------- Request for Scan --------------
@dataclass
class RequestAxialScan:
    id: str
    power_mW: float
    n_measurements: int
    step_size_um: float

# -------------- Scan Result --------------

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

# -------------- Scan Fitting --------------

@dataclass
class FittedAxialScan:
    axial_scan: AxialScan
    fitted_spectras: list[FittedSpectrum]
    fitted_photon_counts: list[PhotonsCounts]

# -------------- Scan Analysis --------------
@dataclass
class AnalysedAxialScan:
    fitted_scan: FittedAxialScan
    freq_shifts: list[AnalyzedFreqShifts]

# -------------- Functions --------------
def fit_axial_scan(scan: AxialScan,
                    do_bg_subtraction: None | bool = None) -> FittedAxialScan:
    spectrum_fitter = SpectrumFitter()

    if do_bg_subtraction is None:
        do_bg_subtraction = scan.system_state.is_do_bg_subtraction_active

    is_reference_mode = scan.system_state.is_reference_mode

    fitted_measurement_points = []
    photons_points = []

    for measurement in scan.measurements:
        frame = measurement.frame_andor.copy()

        if do_bg_subtraction:
            if scan.system_state.bg_image is None:
                raise ValueError(f"No Background image available for this scan")
            frame = subtract_background(frame=frame, bg_frame=scan.system_state.bg_image.mean_image)

        # Generate sline
        sline = spectrum_fitter.get_sline_from_image(frame)

        # Fit spectrum
        fitting = spectrum_fitter.fit(sline=sline, is_reference_mode=is_reference_mode)

        # Photon counts
        photons = calculate_photon_counts_from_fitted_spectrum(fs=fitting,
                                                               preamp_gain=scan.system_state.andor_camera_info.preamp_gain,
                                                               emccd_gain=scan.system_state.andor_camera_info.gain)

        fitted_measurement_points.append(fitting)
        photons_points.append(photons)

        return FittedAxialScan(
            axial_scan=scan,
            fitted_spectras=fitted_measurement_points,
            fitted_photon_counts=photons_points
        )



def analyze_axial_scan(fitted_scan: FittedAxialScan,
                       calibration_calculator: CalibrationCalculator) -> AnalysedAxialScan:
    spectrum_analyzer = SpectrumAnalyzer(calibration_calculator=calibration_calculator)

    analyzed_freqs_shifts = []
    for fitting in fitted_scan.fitted_spectras:
        analyzed_freqs_shifts.append(spectrum_analyzer.analyze_spectrum(fitting))

    return AnalysedAxialScan(
        fitted_scan=fitted_scan,
        freq_shifts=analyzed_freqs_shifts
    )
