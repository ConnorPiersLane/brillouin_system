import numpy as np

from brillouin_system.calibration.calibration import CalibrationData, CalibrationResults, calibrate
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum


def test_calibration_fit():
    # Example known frequencies in GHz
    freqs = [5.10, 5.06, 5.04, 5.02, 5.00]

    # Mock inter-peak distances (pixels)
    inter_peak_px = [36.1, 36.4, 36.5, 36.7, 36.8]

    # Create fake FittedSpectrum objects with only inter-peak distance populated
    mock_spectra = [
        FittedSpectrum(
            is_success=True,
            frame=np.zeros((10, 10)),         # dummy
            sline=np.zeros(10),               # dummy
            x_pixels=np.zeros(10),            # dummy
            fitted_spectrum=np.zeros(10),     # dummy
            x_fit_refined=np.zeros(10),       # dummy
            y_fit_refined=np.zeros(10),       # dummy
            parameters=np.full(7, np.nan),  # not used
            left_peak_center_px=ip+3,
            left_peak_width_px=np.nan,
            left_peak_amplitude=np.nan,
            right_peak_center_px=ip+1,
            right_peak_width_px=np.nan,
            right_peak_amplitude=np.nan,
            inter_peak_distance=ip
        ) for ip in inter_peak_px
    ]

    # Create CalibrationData and run fit
    cal_data = CalibrationData(n_per_freq=1, freqs=freqs, cali_meas_points=[[fs] for fs in mock_spectra])

    result: CalibrationResults = calibrate(cal_data)

    # Evaluate the fit and print coefficients
    print("Quadratic coefficients:")
    print("  a =", result.peak_distance.a)
    print("  b =", result.peak_distance.b)
    print("  c =", result.peak_distance.c)

    # Predict values for plotting or inspection
    test_pixels = np.linspace(min(inter_peak_px), max(inter_peak_px), 100)
    predicted_freqs = result.peak_distance.get_freq(test_pixels)

    # Return for further use (e.g., plotting)
    return test_pixels, predicted_freqs, inter_peak_px, freqs

import matplotlib.pyplot as plt

x_fit, y_fit, x_data, y_data = test_calibration_fit()

plt.figure()
plt.title("Calibration Curve")
plt.plot(x_data, y_data, 'ko', label="Data")
plt.plot(x_fit, y_fit, 'r--', label="Fit")
plt.xlabel("Pixel Distance")
plt.ylabel("Frequency (GHz)")
plt.grid(True)
plt.legend()
plt.show()

