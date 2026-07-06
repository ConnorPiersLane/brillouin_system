import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# =========================
# USER SETTINGS
# =========================
excel_file = "eye_dummy2.xlsx"   # <- change this
sheet_name = 0
output_file = "polar_contour_fixed.png"

angle_col = "Angle"
radius_col = "Radius"
signal_col = "max daq signal [V]"
found_col = "is_found"

# Optional threshold filtering
threshold_col = "threshold_high"
skip_below_threshold = False

# Grid resolution
n_theta = 360
n_r = 250

# =========================
# READ EXCEL
# =========================
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Keep only found rows if available
if found_col in df.columns:
    df = df[df[found_col].astype(str).str.upper().isin(["TRUE", "1", "YES"])]

needed = [angle_col, radius_col, signal_col]
if threshold_col in df.columns:
    needed.append(threshold_col)

df = df[needed].dropna().copy()

# Optional threshold filter
if skip_below_threshold and threshold_col in df.columns:
    df = df[df[signal_col] >= df[threshold_col]]

# =========================
# PREPARE DATA
# =========================
angles_deg = df[angle_col].to_numpy() % 360.0
radii = df[radius_col].to_numpy()
signal = df[signal_col].to_numpy()

angles_rad = np.deg2rad(angles_deg)

# Duplicate data across angular boundaries to avoid wraparound gap
angles_wrap = np.concatenate([
    angles_rad - 2 * np.pi,
    angles_rad,
    angles_rad + 2 * np.pi
])
radii_wrap = np.concatenate([radii, radii, radii])
signal_wrap = np.concatenate([signal, signal, signal])

# Regular polar grid covering full circle
theta_grid = np.linspace(0, 2 * np.pi, n_theta)
r_grid = np.linspace(radii.min(), radii.max(), n_r)
Theta, R = np.meshgrid(theta_grid, r_grid)

# Linear interpolation first
Z = griddata(
    points=(angles_wrap, radii_wrap),
    values=signal_wrap,
    xi=(Theta, R),
    method="linear"
)

# Fill any remaining holes with nearest-neighbor interpolation
Z_nearest = griddata(
    points=(angles_wrap, radii_wrap),
    values=signal_wrap,
    xi=(Theta, R),
    method="nearest"
)

Z = np.where(np.isnan(Z), Z_nearest, Z)

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

contour = ax.contourf(Theta, R, Z, levels=50)
cbar = plt.colorbar(contour, ax=ax, pad=0.1)
cbar.set_label(signal_col)
# contour.set_clim(0, 2)

ax.scatter(
    angles_rad,   # original angles in radians (NOT wrapped)
    radii,
    c='black',
    s=20,
    label="Measurements",
    zorder=5
)

ax.set_title("Polar Contour Plot of Max DAQ Signal")
ax.set_theta_zero_location("E")   # 0° on the right
ax.set_theta_direction(1)         # counterclockwise

plt.tight_layout()
plt.savefig(output_file, dpi=200)
plt.show()