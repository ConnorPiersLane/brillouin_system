import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_measured_circle(json_path: str | Path) -> dict:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fit_plane_basis(points: np.ndarray):
    """
    Returns:
        centroid: (3,)
        e1: in-plane unit vector
        e2: in-plane unit vector
        normal: plane normal unit vector
    """
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid
    _, _, vt = np.linalg.svd(pts_centered, full_matrices=False)

    e1 = vt[0, :]
    e2 = vt[1, :]
    normal = vt[2, :]
    return centroid, e1, e2, normal


def make_circle_points_3d(
    center_3d: np.ndarray,
    radius: float,
    e1: np.ndarray,
    e2: np.ndarray,
    n: int = 300,
) -> np.ndarray:
    thetas = np.linspace(0, 2 * np.pi, n)
    pts = np.array([
        center_3d + radius * np.cos(t) * e1 + radius * np.sin(t) * e2
        for t in thetas
    ])
    return pts


def project_points_to_plane_2d(
    points_3d: np.ndarray,
    origin_3d: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    rel = points_3d - origin_3d
    x = rel @ e1
    y = rel @ e2
    return np.column_stack((x, y))


def main():
    json_path = Path(__file__).parent / "measured_circle.json"
    data = load_measured_circle(json_path)

    # one camera estimate now
    camera_pt = np.array(
        data["camera_estimate_of_pupil_center_xxyyzz"], dtype=float
    )  # shape (3,)
    boundary_pts = np.array(data["reflection_boundary_points_xxyyzz"], dtype=float)
    fitted_center = np.array(
        data["center_fitted_from_reflection_boundary_xxyyzz"], dtype=float
    )
    radius = float(data["radius_fitted_from_reflection_boundary"])
    laser_offset = np.array(data["laser_offset_dxdydz"], dtype=float)

    if len(boundary_pts) < 3:
        raise ValueError("Need at least 3 boundary points to plot fitted circle.")

    centroid, e1, e2, normal = fit_plane_basis(boundary_pts)
    circle_3d = make_circle_points_3d(fitted_center, radius, e1, e2)

    # 2D projection in fitted plane coordinates, centered on fitted center
    boundary_2d = project_points_to_plane_2d(boundary_pts, fitted_center, e1, e2)
    camera_2d = project_points_to_plane_2d(
        camera_pt[np.newaxis, :], fitted_center, e1, e2
    )
    center_2d = np.array([[0.0, 0.0]])
    circle_2d = project_points_to_plane_2d(circle_3d, fitted_center, e1, e2)

    print("Loaded:", json_path)
    print("Number of camera points:   1")
    print(f"Number of boundary points: {len(boundary_pts)}")
    print(f"Camera estimate:           {camera_pt}")
    print(f"Fitted center:             {fitted_center}")
    print(f"Fitted radius:             {radius:.3f}")
    print(f"Laser offset (dx,dy,dz):   {laser_offset}")
    print(f"Plane normal:              {normal}")

    fig = plt.figure(figsize=(12, 10))

    # 3D view
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(
        [camera_pt[0]], [camera_pt[1]], [camera_pt[2]],
        label="Camera estimate",
        marker="o",
        s=50,
    )
    ax3d.scatter(
        boundary_pts[:, 0], boundary_pts[:, 1], boundary_pts[:, 2],
        label="Boundary points",
        marker="^",
        s=40,
    )
    ax3d.scatter(
        [fitted_center[0]], [fitted_center[1]], [fitted_center[2]],
        label="Fitted center",
        marker="x",
        s=100,
    )
    ax3d.plot(
        circle_3d[:, 0], circle_3d[:, 1], circle_3d[:, 2],
        label="Fitted circle",
    )
    ax3d.set_title("3D calibration geometry")
    ax3d.set_xlabel("X [um]")
    ax3d.set_ylabel("Y [um]")
    ax3d.set_zlabel("Z [um]")
    ax3d.legend()

    # XY top-down
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.scatter([camera_pt[0]], [camera_pt[1]], label="Camera estimate", s=50)
    ax_xy.scatter(boundary_pts[:, 0], boundary_pts[:, 1], label="Boundary points", s=40)
    ax_xy.scatter(
        [fitted_center[0]], [fitted_center[1]],
        label="Fitted center",
        marker="x",
        s=100,
    )
    ax_xy.plot(circle_3d[:, 0], circle_3d[:, 1], label="Fitted circle")
    ax_xy.set_title("XY projection")
    ax_xy.set_xlabel("X [um]")
    ax_xy.set_ylabel("Y [um]")
    ax_xy.axis("equal")
    ax_xy.legend()

    # XZ
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.scatter([camera_pt[0]], [camera_pt[2]], label="Camera estimate", s=50)
    ax_xz.scatter(boundary_pts[:, 0], boundary_pts[:, 2], label="Boundary points", s=40)
    ax_xz.scatter(
        [fitted_center[0]], [fitted_center[2]],
        label="Fitted center",
        marker="x",
        s=100,
    )
    ax_xz.plot(circle_3d[:, 0], circle_3d[:, 2], label="Fitted circle")
    ax_xz.set_title("XZ projection")
    ax_xz.set_xlabel("X [um]")
    ax_xz.set_ylabel("Z [um]")
    ax_xz.legend()

    # Plane-local 2D view
    ax_plane = fig.add_subplot(2, 2, 4)
    ax_plane.scatter(camera_2d[:, 0], camera_2d[:, 1], label="Camera estimate", s=50)
    ax_plane.scatter(boundary_2d[:, 0], boundary_2d[:, 1], label="Boundary points", s=40)
    ax_plane.scatter(center_2d[:, 0], center_2d[:, 1], label="Fitted center", marker="x", s=100)
    ax_plane.plot(circle_2d[:, 0], circle_2d[:, 1], label="Fitted circle")
    ax_plane.set_title("Plane-local coordinates")
    ax_plane.set_xlabel("u [um]")
    ax_plane.set_ylabel("v [um]")
    ax_plane.axis("equal")
    ax_plane.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()