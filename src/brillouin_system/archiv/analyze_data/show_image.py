import os

import numpy as np
from matplotlib import pyplot as plt

from brillouinDAQ.analyze_data.load_data import load_measurements
from brillouinDAQ.my_dataclasses.brillouin_viewer_results import OneMeasurementData


def show_fitting(data: list[OneMeasurementData], index: int):
    result = data[index]

    frame = result.fitting_results.frame_andor
    sline = result.fitting_results.sline
    fit = result.fitting_results.fitted_spectrum
    interpeak = result.fitting_results.inter_peak_distance_px
    sd = result.spectrometer_results.spectral_dispersion
    fsr = result.spectrometer_results.free_spectral_range

    row_idx = np.argmax(frame.sum(axis=1))
    x = np.arange(len(sline))

    if sd > 0 and fsr > 0 and not np.isnan(interpeak):
        shift_ghz = 0.5 * fsr - 0.5 * sd * interpeak
        shift_label = f"{shift_ghz:.2f} GHz"
    else:
        shift_label = "- GHz"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    fig.suptitle(f"Brillouin Result #{index}")

    ax1.imshow(frame, cmap='gray', origin='lower', aspect='auto')
    ax1.set_title(f"Camera Frame | Row {row_idx}")
    ax1.set_xlabel("X Pixel")
    ax1.set_ylabel("Y Pixel")

    ax2.plot(x, sline, 'k.', label="Spectrum")
    ax2.plot(x, fit, 'r--', label="Fit")
    ax2.set_title(f"Spectrum Fit | Interpeak: {interpeak:.2f} px / {shift_label}")
    ax2.set_xlabel("Pixel")
    ax2.set_ylabel("Intensity")
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

folder = r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2025-5-6"
file="water.pkl"
file="plastic_water_moved2.pkl"
file_path = os.path.join(folder, file)

data = load_measurements(file_path)
# print_x_axis_is_number(data)
show_fitting(data, 6)