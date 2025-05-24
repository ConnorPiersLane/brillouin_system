import numpy as np
import matplotlib.pyplot as plt

from brillouin_system.my_dataclasses.calibration import (
    CalibrationData,
    CalibrationResults,
    calibrate, calibration_fig,
)
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum



def test_show_calibration_plot():
    # Base inter-peak pixel values for each frequency
    base_inter_peaks = [36.1, 36.4, 36.5, 36.7, 36.8]
    freqs = [5.10, 5.06, 5.04, 5.02, 5.00]

    # Generate 10 noisy FittedSpectrum per frequency
    data = []

    for base_px in base_inter_peaks:
        spectra_per_freq = []
        for _ in range(10):
            noise = np.random.normal(loc=0, scale=0.05)  # small Gaussian noise
            ip = base_px + noise
            fs = FittedSpectrum(
                is_success=True,
                frame=np.zeros((10, 10)),
                sline=np.zeros(10),
                x_pixels=np.zeros(10),
                fitted_spectrum=np.zeros(10),
                x_fit_refined=np.zeros(10),
                y_fit_refined=np.zeros(10),
                lorentzian_parameters=np.full(7, np.nan),
                left_peak_center_px=ip + 3,
                left_peak_width_px=np.nan,
                left_peak_amplitude=np.nan,
                right_peak_center_px=ip + 1,
                right_peak_width_px=np.nan,
                right_peak_amplitude=np.nan,
                inter_peak_distance=ip,
            )
            spectra_per_freq.append(fs)
        data.append(spectra_per_freq)

    # Build CalibrationData object
    cal_data = CalibrationData(n_per_freq=10, freqs=freqs, data=data)

    # Run calibration
    results: CalibrationResults = calibrate(cal_data)

    # Generate plot for 'distance' mode
    fig = calibration_fig(cal_data, results, reference="right")

    # Display the plot
    plt.show()


if __name__ == "__main__":
    test_show_calibration_plot()
