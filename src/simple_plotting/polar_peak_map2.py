import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle

# ==============================
# User data
# ==============================
# Repeated measurement coordinates
# ==============================
# NEW User data
# ==============================

x = np.array([
    0.21, -0.06, -0.06, -0.06, -0.07, -0.07,
     -0.14, -0.14, 0.00, 0.00, 0.00, 0.00,
    -0.04, 0.68, 2.01, 2.54, 0.79, 1.47, -0.12
])

y = np.array([
    -0.65, -0.96, -0.96, -0.96, -0.05, -0.05,
       0.00,  0.00, -0.02,  0.06,  0.06,  0.06,
    -0.04,  0.14,  0.19, -0.11,  1.01,  1.45, -1.94
])

left_peak = np.array([
    5.736, 5.653, 5.736, 5.708, 5.704, 5.664,
    5.683, 5.780, 5.728, 5.663, 5.718,
    5.724, 5.673, 5.813, 5.764, 5.659, 5.665,
    5.718, 5.738
])

right_peak = np.array([
    5.652, 5.685, 5.661, 5.684, 5.573, 5.623,
    5.688, 5.717, 5.689, 5.743, 5.657, 5.678,
    5.641, 5.742, 5.668, 5.626, 5.653, 5.706,
    5.633
])

peak_distance = np.array([
    5.698, 5.668, 5.702, 5.697, 5.645, 5.645,
    5.685, 5.751, 5.710, 5.699, 5.690, 5.703,
    5.658, 5.781, 5.720, 5.644, 5.660, 5.712,
    5.691
])

# ==============================
# Choose what to plot
# ==============================
# Options: "left", "right", "distance"
DATASET = "distance"

OUTER_RADIUS_MM = 3.0
RINGS_MM = [1.0, 2.0, 3.0]

SHOW_POINT_LABELS = False
SAVE_FIGURE = False
OUTPUT_FIG = f"{DATASET}_polar_map.png"

# Flipped colormap
CMAP = "turbo_r"

# Fixed color range and tick spacing
VMIN = 5.64
VMAX = 5.76
LEVELS = np.arange(VMIN, VMAX + 0.001, 0.02)

z_lookup = {
    "left": left_peak,
    "right": right_peak,
    "distance": peak_distance,
}

if DATASET not in z_lookup:
    raise ValueError(f"DATASET must be one of {list(z_lookup.keys())}")

z_raw = z_lookup[DATASET]

# ==============================
# Average repeated measurements at identical (x, y)
# ==============================
points = np.column_stack((x, y))
unique_points, inverse = np.unique(points, axis=0, return_inverse=True)

x_u = unique_points[:, 0]
y_u = unique_points[:, 1]
z_u = np.array([z_raw[inverse == i].mean() for i in range(len(unique_points))])
counts = np.array([np.sum(inverse == i) for i in range(len(unique_points))])

# ==============================
# Build triangulation and mask outside radius
# ==============================
tri = mtri.Triangulation(x_u, y_u)

triangle_centers_x = x_u[tri.triangles].mean(axis=1)
triangle_centers_y = y_u[tri.triangles].mean(axis=1)
triangle_centers_r = np.hypot(triangle_centers_x, triangle_centers_y)

tri.set_mask(triangle_centers_r > OUTER_RADIUS_MM)

# ==============================
# Plot
# ==============================
fig, ax = plt.subplots(figsize=(7, 6))

contour = ax.tricontourf(
    tri,
    z_u,
    levels=200,          # high number = smooth gradient
    cmap="turbo_r",      # flipped colormap
    vmin=5.64,
    vmax=5.76
)

# Optional contour lines
ax.tricontour(
    tri,
    z_u,
    levels=LEVELS,
    colors="k",
    linewidths=0.5,
    alpha=0.35
)

# ==============================
# Overlay circular/polar guide grid
# ==============================
for r in RINGS_MM:
    ring = Circle(
        (0, 0), r,
        fill=False,
        edgecolor="0.35",
        linewidth=0.8,
        zorder=5
    )
    ax.add_patch(ring)

for angle_deg in range(0, 360, 45):
    theta = np.deg2rad(angle_deg)
    ax.plot(
        [0, OUTER_RADIUS_MM * np.cos(theta)],
        [0, OUTER_RADIUS_MM * np.sin(theta)],
        color="0.35",
        linewidth=0.6,
        alpha=0.7,
        zorder=5
    )

# Outer boundary circle
outer = Circle(
    (0, 0), OUTER_RADIUS_MM,
    fill=False,
    edgecolor="0.2",
    linewidth=1.0,
    zorder=6
)
ax.add_patch(outer)

# ==============================
# Show measurement points
# ==============================
ax.scatter(
    x_u, y_u,
    marker="o",
    s=36,
    facecolor="white",
    edgecolor="black",
    linewidth=0.8,
    zorder=7
)

if SHOW_POINT_LABELS:
    for xi, yi, zi in zip(x_u, y_u, z_u):
        ax.text(
            xi, yi + 0.10,
            f"{zi:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            zorder=8
        )

# Optional: show number of repeats at each point
# for xi, yi, n in zip(x_u, y_u, counts):
#     ax.text(xi, yi - 0.12, f"n={n}", ha="center", va="top", fontsize=7, color="black", zorder=8)

# ==============================
# Axes formatting
# ==============================
cbar = fig.colorbar(contour, ax=ax, pad=0.03)
cbar.set_label("Shift [GHz]")
cbar.set_ticks(np.arange(VMIN, VMAX + 0.001, 0.02))

ax.set_aspect("equal")
ax.set_xlim(-OUTER_RADIUS_MM - 0.2, OUTER_RADIUS_MM + 0.2)
ax.set_ylim(-OUTER_RADIUS_MM - 0.2, OUTER_RADIUS_MM + 0.2)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title(f"{DATASET.capitalize()} peak map")

# Label the ring radii on the positive x-axis
for r in RINGS_MM:
    ax.text(
        r, 0.14,
        f"{int(r)} mm",
        fontsize=8,
        color="0.25",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

if SAVE_FIGURE:
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_FIG}")

plt.show()