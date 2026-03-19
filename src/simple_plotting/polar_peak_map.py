import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle

# ==============================
# User data
# ==============================
# Repeated measurement coordinates
x = np.array([
    -0.34, -0.34, -0.34, -0.09, -0.09, -0.09, -0.15, -0.15, -0.15,
     1.02,  1.02,  1.02,  0.73,  0.73, -0.72, -0.72, -0.72,
    -0.99, -0.99, -0.99, -0.64, -0.64, -0.64,
     0.04,  0.04,  0.04,  0.69,  0.69,  0.69,
     2.19,  2.19,  2.19,  1.41,  1.41,  1.41,
     0.04,  0.04,  0.04, -1.24, -1.24, -1.24,
     0.08,  0.08,  0.08
])

y = np.array([
    -0.15, -0.15, -0.15,  0.06,  0.06,  0.06, -0.14, -0.14, -0.14,
     0.03,  0.03,  0.03,  0.77,  0.77,  0.78,  0.78,  0.78,
     0.00,  0.00,  0.00, -0.72, -0.72, -0.72,
    -0.96, -0.96, -0.96, -0.55, -0.55, -0.55,
     0.14,  0.14,  0.14,  1.36,  1.36,  1.36,
     1.97,  1.97,  1.97, -1.39, -1.39, -1.39,
    -1.98, -1.98, -1.98
])

left_peak = np.array([
    5.695, 5.728, 5.610, 5.648, 5.621, 5.665, 5.649, 5.602,
    5.805, 5.533, 5.616, 5.650, 5.678, 5.720, 5.731, 5.718,
    5.734, 5.694, 5.643, 5.667, 5.666, 5.626, 5.645, 5.612,
    5.798, 5.720, 5.696, 5.641, 5.751, 5.617, 5.645, 5.599,
    5.697, 5.647, 5.689, 5.634, 5.701, 5.678, 5.647, 5.637,
    5.568, 5.645, 5.630, 5.762
])

right_peak = np.array([
    5.621, 5.650, 5.694, 5.647, 5.646, 5.667, 5.667, 5.601,
    5.730, 5.659, 5.715, 5.622, 5.687, 5.633, 5.637, 5.649,
    5.767, 5.685, 5.644, 5.696, 5.719, 5.773, 5.588, 5.683,
    5.733, 5.644, 5.651, 5.700, 5.739, 5.704, 5.639, 5.717,
    5.743, 5.694, 5.708, 5.655, 5.639, 5.600, 5.699, 5.642,
    5.600, 5.732, 5.743, 5.727
])

peak_distance = np.array([
    5.661, 5.692, 5.648, 5.647, 5.632, 5.666, 5.657, 5.601,
    5.771, 5.590, 5.661, 5.637, 5.682, 5.680, 5.688, 5.687,
    5.749, 5.690, 5.643, 5.680, 5.690, 5.693, 5.619, 5.644,
    5.769, 5.685, 5.675, 5.667, 5.746, 5.656, 5.642, 5.653,
    5.718, 5.668, 5.697, 5.644, 5.673, 5.643, 5.671, 5.639,
    5.582, 5.684, 5.681, 5.746
])

# ==============================
# Choose what to plot
# ==============================
# Options: "left", "right", "distance"
DATASET = "right"

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