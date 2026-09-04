import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
excel_file = "eye_dummy2.xlsx"   # <- change this
sheet_name = 0                  # sheet index or sheet name
output_file = "daq_signal_vs_radius_by_angle.png"

angle_col = "Angle"
radius_col = "Radius"
signal_col = "max daq signal [V]"
found_col = "is_found"
threshold_col = "threshold_high"

# Set to True if you want to exclude points below threshold
skip_below_threshold = True

# =========================
# READ EXCEL
# =========================
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Keep only found rows if column exists
if found_col in df.columns:
    df = df[df[found_col].astype(str).str.upper().isin(["TRUE", "1", "YES"])]

# Keep only needed columns
needed = [angle_col, radius_col, signal_col]
if threshold_col in df.columns:
    needed.append(threshold_col)

df = df[needed].dropna().copy()

# =========================
# ANGLE PROCESSING
# =========================
# Convert angles to 0..360
df[angle_col] = df[angle_col] % 360

# Round to nearest 45 degrees
df["Angle_group"] = (np.round(df[angle_col] / 45) * 45) % 360
df["Angle_group"] = df["Angle_group"].astype(int)

# =========================
# OPTIONAL THRESHOLD FILTER
# =========================
threshold_value = None
if threshold_col in df.columns:
    threshold_series = pd.to_numeric(df[threshold_col], errors="coerce")
    threshold_value = float(threshold_series.median())

    if skip_below_threshold:
        df = df[df[signal_col] >= threshold_series]

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<']
angle_groups = sorted(df["Angle_group"].unique())

for i, ang in enumerate(angle_groups):
    sub = df[df["Angle_group"] == ang]
    ax.scatter(
        sub[radius_col],
        sub[signal_col],
        s=60,
        marker=markers[i % len(markers)],
        label=f"{ang}°"
    )

# Threshold line
if threshold_value is not None:
    ax.axhline(
        threshold_value,
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold_value:.4f})"
    )

ax.set_title("DAQ Signal vs Radius (by Angle)")
ax.set_xlabel("Radius")
ax.set_ylabel("Max DAQ Signal [V]")
ax.grid(True, alpha=0.5)
ax.legend(title="Angle group", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig(output_file, dpi=200)
plt.show()