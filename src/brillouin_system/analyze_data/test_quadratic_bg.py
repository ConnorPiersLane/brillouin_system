# Fit and plot
import pickle
import numpy as np
import matplotlib.pyplot as plt

from brillouin_system.utils.fit_spectrum_quadratic_bg import (
    fit_lorentzian_peaks_with_quadratic_bg,
    _fit_background_quadratic,
    _quadratic,
    _spectrum_function_quadratic_bg
)
from brillouin_system.utils.fit_util import refine_fitted_spectrum

# Data
sline = np.array([
    837.6, 835.4, 830., 830.9, 827.3, 826.7, 841.3, 854.7,
    907.7, 1492.1, 3276.9, 3531.1, 3312.4, 3044.7, 3035.2, 3111.9,
    3210.4, 3479.3, 3686.9, 3776., 3877.4, 3931.9, 4014.9, 4076.4,
    4143.5, 4167.7, 4181.5, 4153.4, 4289., 4245.9, 4384.5, 4206.7,
    4162.5, 4135.3, 4127.6, 4082.3, 4092.3, 4128.1, 4257.3, 4358.9,
    4458.7, 4480., 4511.2, 4567.4, 4732., 4669.1, 4610., 4647.3,
    4592.6, 4496., 4601.8, 4612.8, 4486.8, 4631.3, 4520.5, 4567.4,
    4508.5, 4664.9, 4548.4, 4570.7, 4437.2, 4485.9, 4481., 4551.3,
    4717.8, 4818.4, 4583., 2035.1, 1025., 877.7, 858.8, 868.2,
    844.1, 847.5, 856.2, 845.5, 854.3, 851.6, 854.3, 855.
])
px = np.arange(len(sline))

# Load series1.pkl
series1_path = "2025-5-26/series4.pkl"
with open(series1_path, "rb") as f:
    series_list = pickle.load(f)

series = series_list[1]

# Extract signal sline from the first measurement
sline_signal = series.measurements[2].fitting_results.sline


# Combine background and signal
sline_total = sline + sline_signal

# Fit the background
popt_bg = _fit_background_quadratic(px, sline_total)
spectrum_bg = _quadratic(px, *popt_bg)

# Fit the combined spectrum
popt = fit_lorentzian_peaks_with_quadratic_bg(sline_total)
# Compute the total fitted model (Lorentzian + background)
fitted_total = _spectrum_function_quadratic_bg(px, *popt)

# Print fitted parameters
param_names = [
    'amp1', 'cen1', 'wid1',
    'amp2', 'cen2', 'wid2',
    'a', 'b', 'c'
]
print("\nFitted Parameters:")
for name, value in zip(param_names, popt):
    print(f"{name} = {value:.4f}")

x_plot, y_plot = refine_fitted_spectrum(_spectrum_function_quadratic_bg, px, popt, factor=10)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(px, sline_signal, 'k.', label='Measured Sline')
plt.plot(px, sline_total, 'b.', label='Measured Sline + Background')
plt.plot(x_plot, y_plot, 'r-', label='Total Fit (Lorentzian + Background)')
plt.xlabel('Pixel')
plt.ylabel('Intensity')
plt.title('S-Line Fit: Lorentzian Peaks + Quadratic Background')
plt.legend()
plt.grid(True)
plt.show()
