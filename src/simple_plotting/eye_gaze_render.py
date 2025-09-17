import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

def make_colored_sphere(radius=7.7):
    """Return a PyVista sphere mesh with white/grey/blue caps."""
    sphere = pv.Sphere(radius=radius, theta_resolution=60, phi_resolution=60)
    points = sphere.points
    r = np.linalg.norm(points, axis=1)
    phi = np.arccos(points[:, 2] / r)  # polar angle from +Z

    colors = np.full((points.shape[0], 3), [0.6, 0.8, 1.0])  # light blue
    colors[phi <= np.deg2rad(50)] = [0.7, 0.7, 0.7]  # grey outer cap
    colors[phi <= np.deg2rad(10)] = [1.0, 1.0, 1.0]  # white inner cap
    sphere["colors"] = colors
    return sphere

def render_eye(yaw_deg=90, pitch_deg=0, radius=7.7):
    """Render the sphere from a rotated camera view and return as numpy image."""
    sphere = make_colored_sphere(radius)
    plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
    plotter.add_mesh(sphere, scalars="colors", rgb=True, smooth_shading=True)

    # Compute camera direction vector
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    direction = np.array([
        np.cos(pitch) * np.sin(yaw),
        -np.sin(pitch),
        -np.cos(pitch) * np.cos(yaw)
    ])
    plotter.camera_position = [(2, 0, 30), direction, (0, 1, 0)]

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img

# Example usage: sweep yaw and pitch and plot last frame
if __name__ == "__main__":
    img = render_eye(yaw_deg=40, pitch_deg=0)

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
