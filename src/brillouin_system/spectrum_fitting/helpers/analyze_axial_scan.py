import numpy as np

from brillouin_system.calibration.calibration import CalibrationCalculator

from brillouin_system.my_dataclasses.human_interface_measurements import AxialScan, AnalyzedAxialScan, \
    AnalyzedMeasurementPoint
from brillouin_system.spectrum_fitting.spectrum_analyzer import SpectrumAnalyzer
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter
from brillouin_system.spectrum_fitting.helpers.calculate_photon_counts import calculate_photon_counts_from_fitted_spectrum


def analyze_axial_scan(scan: AxialScan,
                       calibration_calculator: CalibrationCalculator,
                       do_bg_subtraction) -> AnalyzedAxialScan:
    spectrum_fitter = SpectrumFitter()
    spectrum_analyzer = SpectrumAnalyzer(calibration_calculator=calibration_calculator)

    is_reference_mode = scan.system_state.is_reference_mode
    do_bg_subtraction = do_bg_subtraction


    analyzed_measurements = []

    for measurement in scan.measurements:
        frame = measurement.frame_andor.copy()

        if do_bg_subtraction:
            if scan.system_state.bg_image is None:
                raise ValueError(f"No Background image available for this scan")
            frame = frame - scan.system_state.bg_image.mean_image
            frame = np.clip(frame, 0, None)

        # Generate sline
        sline = spectrum_fitter.get_sline_from_image(frame)

        # Fit spectrum
        fitting = spectrum_fitter.fit(sline=sline, is_reference_mode=is_reference_mode)

        # Analyze fitted spectrum
        freq_shifts = spectrum_analyzer.analyze_spectrum(fitting)

        # Photon counts
        photons = calculate_photon_counts_from_fitted_spectrum(fs=fitting,
                                                               preamp_gain=scan.system_state.andor_camera_info.preamp_gain,
                                                               emccd_gain=scan.system_state.andor_camera_info.gain)

        analyzed_measurements.append(
            AnalyzedMeasurementPoint(
                fitted_spectrum=fitting,
                freq_shifts=freq_shifts,
                photons=photons,
            )
        )

        return AnalyzedAxialScan(axial_scan=scan,
                                 analyzed_measurements=analyzed_measurements)
