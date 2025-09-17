import numpy as np
import matplotlib.pyplot as plt

def plot_sphere_two_caps(radius=1.0, spacing_deg=10, cap1_deg=10, cap2_deg=50):
    """
    Generate and plot a sphere with two concentric caps:
    0..cap1 = white, cap1..cap2 = grey, rest = light blue.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    spacing_deg : float
        Angular step in degrees for phi and theta.
    cap1_deg : float
        Inner cap angle in degrees (white).
    cap2_deg : float
        Outer cap angle in degrees (grey).
    """
    # Convert angles to radians
    spacing = np.deg2rad(spacing_deg)
    cap1 = np.deg2rad(cap1_deg)
    cap2 = np.deg2rad(cap2_deg)

    # Create spherical grid
    phi = np.arange(0, np.pi + spacing, spacing)       # polar angle (0..π)
    theta = np.arange(0, 2*np.pi + spacing, spacing)   # azimuth (0..2π)
    phi, theta = np.meshgrid(phi, theta)

    # Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Base color: light blue
    colors = np.full(x.shape + (3,), [0.6, 0.8, 1.0])  # RGB light blue

    # Grey cap region (0..cap2)
    cap2_mask = phi <= cap2
    colors[cap2_mask] = [0.7, 0.7, 0.7]  # grey

    # White inner cap (0..cap1)
    cap1_mask = phi <= cap1
    colors[cap1_mask] = [1.0, 1.0, 1.0]  # white

    # Plotting
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, facecolors=colors, edgecolor='k', linewidth=0.5, shade=False)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=80, azim=0)  # nice 3D view
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_sphere_two_caps(radius=7.7, spacing_deg=10, cap1_deg=5, cap2_deg=50)
