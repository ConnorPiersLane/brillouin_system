import os


import numpy as np
from matplotlib import pyplot as plt

from brillouinDAQ.analyze_data.load_data import load_measurements


def print_x_axis_is_number(data):
    # Extract frequency shift values
    freq_shifts = [entry.freq_shift_ghz for entry in data]

    # Compute mean and standard deviation
    mean_shift = np.mean(freq_shifts)
    std_shift = np.std(freq_shifts)

    # Generate x-axis values
    x = np.arange(len(freq_shifts))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, freq_shifts, marker='o', linestyle='-', label='Frequency Shift (GHz)')
    plt.axhline(mean_shift, color='r', linestyle='--', label=f'Mean = {mean_shift:.3f} GHz')
    plt.fill_between(x, mean_shift - std_shift, mean_shift + std_shift, color='gray', alpha=0.3,
                     label=f'±1 Std Dev ({std_shift:.3f} GHz)')

    # Formatting
    plt.ylim(4.9, 5.1)
    plt.xlabel('Index in List')
    plt.ylabel('Frequency Shift (GHz)')
    plt.title(
        f'Brillouin Frequency Shift. Expo: {round(data[0].camera_settings_ready.exposure_time_s, ndigits=2)}, Gain: {data[0].camera_settings_ready.gain}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_x_zaber(data):
    # Extract frequency shift values
    freq_shifts = [entry.freq_shift_ghz for entry in data]


    # Generate x-axis values
    x0 = data[0].zaber_position.x
    x = [entry.zaber_position.x - x0 for entry in data]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, freq_shifts, marker='o', linestyle='-', label='Frequency Shift (GHz)')

    # Formatting

    plt.xlabel('Lens Position (um)')
    plt.ylabel('Frequency Shift (GHz)')
    plt.title(f'Brillouin Frequency Shift. Expo: {round(data[0].camera_settings_ready.exposure_time_s, ndigits=2)}, Gain: {data[0].camera_settings_ready.gain}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



folder = r"C:\Users\cplan\Partners HealthCare Dropbox\Connor Lane\Data\2025-5-6"
file="water.pkl"
file="plastic_water_moved2.pkl"
file_path = os.path.join(folder, file)

data = load_measurements(file_path)
# print_x_axis_is_number(data)
print_x_zaber(data)