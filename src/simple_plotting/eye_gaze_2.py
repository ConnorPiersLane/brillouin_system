import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(yaw_deg=0, pitch_deg=0):
    """Return a rotation matrix for yaw (around Y) and pitch (around X)."""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [ 0,           1, 0          ],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    return Rx @ Ry

def plot_sphere_two_caps(radius=1.0, spacing_deg=10, cap1_deg=10, cap2_deg=50,
                         yaw_deg=0, pitch_deg=0):
    """
    Plot a sphere with a white inner cap (0..cap1) and grey outer cap (cap1..cap2),
    rotated by yaw/pitch so that the cap is actually rotated on the sphere surface.
    """
    spacing = np.deg2rad(spacing_deg)
    cap1 = np.deg2rad(cap1_deg)
    cap2 = np.deg2rad(cap2_deg)

    # Generate sphere mesh (unrotated)
    phi = np.arange(0, np.pi + spacing, spacing)
    theta = np.arange(0, 2*np.pi + spacing, spacing)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Compute angle from original +Z axis (phi)
    angle_from_axis = phi  # already polar angle from +Z

    # Base colors
    colors = np.full(x.shape + (3,), [0.6, 0.8, 1.0])  # light blue
    cap2_mask = angle_from_axis <= cap2
    colors[cap2_mask] = [0.7, 0.7, 0.7]
    cap1_mask = angle_from_axis <= cap1
    colors[cap1_mask] = [1.0, 1.0, 1.0]

    # Rotate geometry (so cap moves visually)
    R = rotation_matrix(yaw_deg, pitch_deg)
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rotated = pts @ R.T
    x_rot = rotated[:, 0].reshape(x.shape)
    y_rot = rotated[:, 1].reshape(y.shape)
    z_rot = rotated[:, 2].reshape(z.shape)

    # Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_rot, y_rot, z_rot, facecolors=colors,
                    edgecolor='k', linewidth=0.0, shade=False)

    ax.set_box_aspect([1, 1, 1])
    # ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_axis_off()
    ax.view_init(elev=10, azim=10)  # look straight toward +X
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_sphere_two_caps(radius=7.7, spacing_deg=10, cap1_deg=5, cap2_deg=50,
                         yaw_deg=90, pitch_deg=0)
