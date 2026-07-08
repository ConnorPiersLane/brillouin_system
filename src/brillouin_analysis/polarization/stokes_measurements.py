import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# USER INPUT
# =========================

# Polarizer angles in degrees
angles_deg = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])

# Measured intensities
# calibration: after calibration port [mW]
# intensities = np.array([1.04, 1.23, 0.83, 0.23, 12.2e-3, 0.407, 1.019, 1.25, 0.86, 0.24, 9.5e-3, 0.382, 1.001])
#
# # after sample mirror
# # bad fiber
# # intensities = np.array([3.0, 3.45, 2.26, 0.54, 0.05, 1.25, 2.9, 3.45, 2.26, 0.54, 0.05, 1.25, 2.9])
# intensities = np.array([2.9, 3.3, 3.1, 0.5, 0.05, 1.2, 2.8, 3.3, 3.1, 0.5, 0.05, 1.2, 2.8])
# # good
# intensities = np.array([0.9, 1.08, 0.72, 0.19, 0.02, 0.38, 0.93, 1.09, 0.72, 0.19, 0.03, 0.39, 0.92])
# intensities = np.array([1.0, 1.15, 0.75, 0.2, 0.04, 0.45, 0.99, 1.15, 0.75, 0.2, 0.04, 0.42, 0.98])
# intensities = np.array([1.1, 1.25, 0.78, 0.19, 0.06, 0.52, 1.12, 1.23, 0.78, 0.19, 0.07, 0.52, 1.10])
#
#
# # # calibration input, before rb filter
# intensities = np.array([6, 5.58, 3.85, 1.8, 0.8, 1.65, 4.15, 6.25, 5.6, 2.9, 0.3, 0.65, 3.9])
# intensities = np.array([4, 6.3, 4.8, 1.9, 0.6, 2.4, 5.0, 6.1, 3.8, 1.3, 1.0, 3.5, 5.9])
# intensities = np.array([5.9, 5.4, 2.1, 0.3, 1.7, 5.0, 6.7, 4.9, 1.5, 0.1, 2.1, 5.3, 6.7])
# intensities = np.array([6.5, 5.0, 2.4, 1.3, 2.3, 4.6, 5.6, 5.2, 3.1, 1.7, 2.0, 3.4, 5.0])


# #
# # sample input, before rb filter
# intensities = np.array([0.27, 0.27, 0.14, 0.02, 0.02, 0.14, 0.27, 0.27, 0.14, 0.02, 0.02, 0.14, 0.26])
intensities = np.array([0.26, 0.27, 0.16, 0.03, 0.02, 0.12, 0.25, 0.27, 0.15, 0.03, 0.02, 0.13, 0.25])


# Correction for angular misalignment, in degrees
# Positive means your measured angle scale is shifted high by this amount.
angle_offset_deg = 25.522 - 90

# Optional: manually enter S3 if measured with quarter-wave plate.
# If unknown, leave as None. Then ellipse is assumed linear/partial-linear only.
S3_manual = None

# Optional correction for waveplate fast-axis offset, degrees
# Only matters if you use S3_manual from a QWP measurement.
fast_axis_offset_deg = 0.0


# =========================
# FIT STOKES PARAMETERS
# =========================

def intensity_model(theta_rad, A, B, C):
    return A + B * np.cos(2 * theta_rad) + C * np.sin(2 * theta_rad)

theta_corr_deg = angles_deg - angle_offset_deg
theta_rad = np.deg2rad(theta_corr_deg)

popt, _ = curve_fit(intensity_model, theta_rad, intensities)
A, B, C = popt

S0 = 2 * A
S1 = 2 * B
S2 = 2 * C
S3 = 0.0 if S3_manual is None else S3_manual

# Normalize Stokes parameters
s1 = S1 / S0
s2 = S2 / S0
s3 = S3 / S0

DoLP = np.sqrt(s1**2 + s2**2)
DoP = np.sqrt(s1**2 + s2**2 + s3**2)

# Polarization azimuth angle phi
phi_rad = 0.5 * np.arctan2(S2, S1)
phi_deg = np.rad2deg(phi_rad)

# Ellipticity angle chi
# chi = 0 for linear, +/-45 deg for circular
chi_rad = 0.5 * np.arcsin(np.clip(s3 / max(DoP, 1e-12), -1, 1))
chi_deg = np.rad2deg(chi_rad)

# Ellipse axis ratio b/a
axis_ratio = abs(np.tan(chi_rad))


# =========================
# PRINT RESULTS
# =========================

print("Corrected angles:", theta_corr_deg)
print()
print("Stokes parameters:")
print(f"S0 = {S0:.6f}")
print(f"S1 = {S1:.6f}")
print(f"S2 = {S2:.6f}")
print(f"S3 = {S3:.6f}")
print()
print("Normalized Stokes:")
print(f"s1 = {s1:.6f}")
print(f"s2 = {s2:.6f}")
print(f"s3 = {s3:.6f}")
print()
print(f"Polarization angle phi = {phi_deg:.3f} deg")
print(f"Degree of linear polarization DoLP = {DoLP:.4f}")
print(f"Degree of polarization DoP = {DoP:.4f}")
print(f"Ellipticity angle chi = {chi_deg:.3f} deg")
print(f"Ellipse axis ratio b/a = {axis_ratio:.4f}")


# =========================
# PLOT FIT
# =========================

theta_fit_deg = np.linspace(0, 360, 500)
theta_fit_rad = np.deg2rad(theta_fit_deg)

I_fit = intensity_model(theta_fit_rad, A, B, C)

# =========================
# COMBINED PLOTS
# =========================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---------------------------------
# LEFT: Rotating polarizer fit
# ---------------------------------

theta_fit_deg = np.linspace(0, 360, 500)
theta_fit_rad = np.deg2rad(theta_fit_deg)

I_fit = intensity_model(theta_fit_rad, A, B, C)

axes[0].scatter(theta_corr_deg, intensities, label="Measured data")
axes[0].plot(theta_fit_deg, I_fit, label="Stokes fit")

axes[0].set_xlabel("Corrected polarizer angle [deg]")
axes[0].set_ylabel("Intensity")
axes[0].set_title("Rotating polarizer fit")
axes[0].legend()
axes[0].grid(True)

# ---------------------------------
# RIGHT: Polarization ellipse
# ---------------------------------

t = np.linspace(0, 2 * np.pi, 1000)

a = 1.0
b = axis_ratio

# Base ellipse
x = a * np.cos(t)
y = b * np.sin(t)

# Rotate by phi
x_rot = x * np.cos(phi_rad) - y * np.sin(phi_rad)
y_rot = x * np.sin(phi_rad) + y * np.cos(phi_rad)

axes[1].plot(x_rot, y_rot)

axes[1].axhline(0, linewidth=0.8)
axes[1].axvline(0, linewidth=0.8)

axes[1].set_aspect("equal", adjustable="datalim")

# Zoom into small ellipticity
margin = 0.1

axes[1].set_xlim(-1.1, 1.1)

ymax = max(abs(y_rot)) + margin
ymax = max(ymax, 0.1)   # minimum visible height

axes[1].set_ylim(-ymax, ymax)
axes[1].set_xlabel("Ex")
axes[1].set_ylabel("Ey")

axes[1].set_title(
    f"Polarization ellipse\n"
    f"phi = {phi_deg:.2f} deg\n"
    f"DoLP = {DoLP:.3f}, DoP = {DoP:.3f}"
)

axes[1].grid(True)

plt.tight_layout()
plt.show()