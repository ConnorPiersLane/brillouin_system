import numpy as np
import pyvista as pv
import math

# ---------- Geometry helpers ----------

import numpy as np
from matplotlib import pyplot as plt


def rotation_matrix_from_gaze(gaze_vector, up_hint=np.array([0, 1, 0])):
    """Return 3×3 rotation matrix aligning +Z with gaze_vector."""
    forward = gaze_vector / np.linalg.norm(gaze_vector)
    right = np.cross(up_hint, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    R = np.eye(3)
    R[:, 0] = right
    R[:, 1] = up
    R[:, 2] = forward
    return R

def apply_pose_to_eye(parts, C, gaze_vector):
    """Apply rigid transform to all eye meshes so cornea center = C, axis = gaze_vector."""
    R = rotation_matrix_from_gaze(gaze_vector)
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = C

    for key in ["globe", "cornea", "iris", "pupil"]:
        parts[key].transform(transform, inplace=True)
    return parts


def cornea_center_from_delta(R_eye, R_cornea, delta_apex):
    """
    Given eyeball radius, cornea radius, and apex protrusion (Δ_apex),
    return (z_cornea_center, z_apex).
    Conventions: globe center at z=0, eye looks along +Z.
    """
    zC = (R_eye + delta_apex) - R_cornea
    z_apex = zC + R_cornea
    return zC, z_apex

def limbus_plane_and_radius(R_eye, R_cornea, zC):
    """
    For two coaxial spheres (globe center at 0, cornea center at zC along +Z),
    return (z_limbus_plane, r_limbus).
    """
    # Plane where both spheres intersect (perpendicular to the axis)
    z_plane = (R_eye**2 - R_cornea**2 + zC**2) / (2.0 * zC)
    # Circle radius (use either sphere; both give the same)
    r_limbus = math.sqrt(max(R_eye**2 - z_plane**2, 0.0))
    return z_plane, r_limbus

def cornea_center_from_target_limbus(R_eye, R_cornea, r_limbus):
    """
    Solve for cornea center zC given a desired limbus radius.
    Returns (zC, z_limbus_plane, delta_apex, z_apex).
    """
    # limbus plane position from globe sphere
    z_plane = math.sqrt(max(R_eye**2 - r_limbus**2, 0.0))
    # Solve quadratic: zC = z_plane ± sqrt(R_cornea^2 - r_limbus^2)
    term = math.sqrt(max(R_cornea**2 - r_limbus**2, 0.0))
    # Pick the physically sensible root (small positive zC, not the huge one)
    zC1 = z_plane + term
    zC2 = z_plane - term
    zC = zC2 if 0 < zC2 < zC1 else zC1
    z_apex = zC + R_cornea
    delta_apex = (z_apex - R_eye)  # how far apex is in front of globe surface
    return zC, z_plane, delta_apex, z_apex

def iris_plane_from_acd(z_apex, ACD):
    """Iris plane is ACD behind the corneal apex along -Z."""
    return z_apex - ACD

# ---------- Mesh builders ----------

def build_clipped_globe(R_eye, z_clip):
    """
    Build globe sphere and clip everything with z > z_clip (keep posterior part).
    """
    globe = pv.Sphere(radius=R_eye, theta_resolution=180, phi_resolution=180)
    globe = globe.clip(normal=(0, 0, 1), origin=(0, 0, z_clip), invert=True)
    return globe

def build_trimmed_cornea_shell(R_cornea, zC, z_limbus, opacity=0.15):
    """
    Build the anterior cornea sphere, translate its center to zC, and
    clip everything with z < z_limbus (keep anterior part).
    """
    cornea = pv.Sphere(radius=R_cornea, theta_resolution=180, phi_resolution=180)
    cornea.translate((0, 0, zC), inplace=True)
    # keep anterior (z >= z_limbus)
    cornea = cornea.clip(normal=(0, 0, -1), origin=(0, 0, z_limbus), invert=True)
    # optional: you can return the actor setup params; styling is applied in renderer
    return cornea

def build_iris_annulus(r_outer, r_inner, z_iris):
    """
    Iris as a flat annulus (outer radius = limbus, inner = pupil) at z_iris.
    """
    iris = pv.Disc(inner=r_inner, outer=r_outer, c_res=256)
    iris.translate((0, 0, z_iris), inplace=True)
    return iris


def build_iris_and_pupil(r_outer, r_inner, z_iris):
    """
    Build iris annulus (outer radius = limbus, inner = pupil radius)
    and a black pupil disk.
    """
    iris = pv.Disc(inner=r_inner, outer=r_outer, c_res=256)
    iris.translate((0, 0, z_iris), inplace=True)
    iris["colors"] = np.tile([0.2, 0.4, 0.2], (iris.n_points, 1))

    pupil = pv.Disc(inner=0.0, outer=r_inner, c_res=128)
    pupil.translate((0, 0, z_iris + 0.001), inplace=True)  # tiny offset to avoid z-fighting
    pupil["colors"] = np.tile([0.0, 0.0, 0.0], (pupil.n_points, 1))
    return iris, pupil


def add_laser_cross(plotter, size=4.0, margin=1.0):
    """
    Draw a cross that is always visible by placing it just in front of the camera.
    Works in orthographic mode (no size change with depth).
    """
    cam = plotter.camera
    # put cross a little in front of the camera along its viewing direction
    # camera looks from position -> focal_point, so we place the cross at z_cam - margin
    # assuming camera looks along -Z toward the scene (your setup)
    z_cam = cam.position[2]
    z_pos = z_cam - abs(margin)

    half = size / 2.0
    cross_x = pv.Line((-half, 0, z_pos), (half, 0, z_pos))
    cross_y = pv.Line((0, -half, z_pos), (0, half, z_pos))

    # tubes render much more reliably than bare GL lines across platforms
    plotter.add_mesh(cross_x, color="red", line_width=3, render_lines_as_tubes=True)
    plotter.add_mesh(cross_y, color="red", line_width=3, render_lines_as_tubes=True)





# ---------- High-level constructor ----------


def create_anatomical_eye(
    R_eye=12.0,
    R_cornea=7.7,
    ACD=3.0,
    *,
    delta_apex=None,
    target_iris_outer_radius=None,
    pupil_radius=2.0,
    iris_margin=0.0,
):
    """
    Build an anatomically consistent eye with globe, cornea, iris annulus, and black pupil.
    """
    if (delta_apex is None) == (target_iris_outer_radius is None):
        raise ValueError("Specify exactly one of delta_apex OR target_iris_outer_radius.")

    if delta_apex is not None:
        zC, z_apex = cornea_center_from_delta(R_eye, R_cornea, delta_apex)
        z_limbus, r_limbus = limbus_plane_and_radius(R_eye, R_cornea, zC)
    else:
        zC, z_limbus, delta_apex, z_apex = cornea_center_from_target_limbus(
            R_eye, R_cornea, target_iris_outer_radius
        )
        r_limbus = target_iris_outer_radius

    z_iris = iris_plane_from_acd(z_apex, ACD)

    # Build meshes
    globe = build_clipped_globe(R_eye, z_clip=z_limbus)
    cornea = build_trimmed_cornea_shell(R_cornea, zC, z_limbus)
    iris, pupil = build_iris_and_pupil(max(r_limbus - iris_margin, 0.0), pupil_radius, z_iris)

    globe["colors"] = np.tile([0.93, 0.93, 0.95], (globe.n_points, 1))

    return dict(
        globe=globe,
        cornea=cornea,
        iris=iris,
        pupil=pupil,
        z_apex=z_apex,
        z_limbus=z_limbus,
        r_limbus=r_limbus,
        z_iris=z_iris,
        z_cornea_center=zC,
        delta_apex=delta_apex,
    )

# ---------- Minimal renderer example (optional) ----------

def demo_render():
    parts = create_anatomical_eye(
        R_eye=12.0,
        R_cornea=7.7,
        ACD=3.0,
        target_iris_outer_radius=6.0,
        pupil_radius=2.5,
    )

    plotter = pv.Plotter(window_size=(600, 600))
    plotter.enable_parallel_projection()
    cam = plotter.camera
    cam.focal_point = (0, 0, parts["z_limbus"])
    cam.position = (0, 0, 60)
    cam.SetViewUp(0, 1, 0)
    cam.parallel_scale = 15
    cam.clipping_range = (0.1, 200)

    plotter.add_mesh(parts["globe"], scalars="colors", rgb=True, smooth_shading=True)
    plotter.add_mesh(parts["cornea"], color="white", opacity=0.5,  # ✅ less transparent
                     smooth_shading=True, specular=1.0, specular_power=50)
    plotter.add_mesh(parts["iris"], scalars="colors", rgb=True, smooth_shading=True)
    plotter.add_mesh(parts["pupil"], scalars="colors", rgb=True, smooth_shading=False)
    add_laser_cross(plotter, size=4.0, z=parts["z_limbus"])


    plotter.show()

def demo_render_with_pose():
    # Build anatomical eye (still at default origin, aligned along +Z)
    parts = create_anatomical_eye(
        R_eye=12.0,
        R_cornea=7.7,
        ACD=3.0,
        target_iris_outer_radius=6.0,
        pupil_radius=2.5,
    )

    # --- Define desired pose ---
    C = np.array([2.0, 1.0, 0.0])  # example cornea center position
    gaze = np.array([0.1, 0.2, 1.0])  # arbitrary gaze direction
    gaze /= np.linalg.norm(gaze)

    # Apply rigid transform
    apply_pose_to_eye(parts, C, gaze)

    # --- Plotting ---
    plotter = pv.Plotter(window_size=(600, 600))
    plotter.enable_parallel_projection()
    cam = plotter.camera
    cam.focal_point = (0, 0, C[2])
    cam.position = (0, 0, 60)
    cam.SetViewUp(0, 1, 0)
    cam.parallel_scale = 15

    # Add meshes
    plotter.add_mesh(parts["globe"], scalars="colors", rgb=True, smooth_shading=True)
    plotter.add_mesh(parts["cornea"], color="white", opacity=0.3,
                     smooth_shading=True, specular=1.0, specular_power=50)
    plotter.add_mesh(parts["iris"], scalars="colors", rgb=True, smooth_shading=True)
    plotter.add_mesh(parts["pupil"], scalars="colors", rgb=True)

    # Add fixed laser cross
    add_laser_cross(plotter, size=4.0, margin=1.0)

    plotter.show()



class EyeSceneRenderer:
    def __init__(self,
                 R_eye=12.0,
                 R_cornea=7.7,
                 ACD=3.0,
                 target_iris_outer_radius=6.0,
                 pupil_radius=2.5,
                 window_size=(512, 512)):
        """Initialize scene with globe, cornea, iris, pupil, and laser cross."""
        self.parts = create_anatomical_eye(
            R_eye=R_eye,
            R_cornea=R_cornea,
            ACD=ACD,
            target_iris_outer_radius=target_iris_outer_radius,
            pupil_radius=pupil_radius,
        )

        # Keep untransformed originals for later resets
        self.original_meshes = {k: v.copy() for k, v in self.parts.items()
                                if k in ("globe", "cornea", "iris", "pupil")}

        self.plotter = pv.Plotter(off_screen=True, window_size=window_size)
        self.plotter.enable_parallel_projection()

        # --- Camera setup ---
        cam = self.plotter.camera
        cam.focal_point = (0, 0, self.parts["z_limbus"])
        cam.position = (0, 0, 60)
        cam.SetViewUp(0, 1, 0)
        cam.parallel_scale = 15
        cam.clipping_range = (0.1, 1000)

        # --- Add actors once ---
        self.actors = {}
        self.actors["globe"] = self.plotter.add_mesh(
            self.parts["globe"], scalars="colors", rgb=True, smooth_shading=True)
        self.actors["cornea"] = self.plotter.add_mesh(
            self.parts["cornea"], color="white", opacity=0.3,
            smooth_shading=True, specular=1.0, specular_power=50)
        self.actors["iris"] = self.plotter.add_mesh(
            self.parts["iris"], scalars="colors", rgb=True, smooth_shading=True)
        self.actors["pupil"] = self.plotter.add_mesh(
            self.parts["pupil"], scalars="colors", rgb=True)

        add_laser_cross(self.plotter, size=4.0, margin=1.0)

    def set_pupil_diameter(self, new_diameter):
        """Regenerate pupil geometry and update actor in place."""
        r_outer = self.parts["r_limbus"]  # keep iris outer radius constant
        r_inner = new_diameter / 2.0

        # Build new pupil (and keep iris plane position)
        z_iris = self.parts["z_iris"]
        new_pupil = pv.Disc(inner=0.0, outer=r_inner, c_res=128)
        new_pupil.translate((0, 0, z_iris + 0.001), inplace=True)
        new_pupil["colors"] = np.tile([0.0, 0.0, 0.0], (new_pupil.n_points, 1))

        # Store and update actor
        self.parts["pupil"].deep_copy(new_pupil)
        self.actors["pupil"].mapper.SetInputData(self.parts["pupil"])
        self.plotter.render()

    def set_eye_pose(self, C, gaze_vector):
        """Update mesh transforms without rebuilding geometry."""
        # Reset meshes to original (important for repeated updates)
        for key, orig_mesh in self.original_meshes.items():
            self.parts[key].deep_copy(orig_mesh)

        # Apply new transform
        apply_pose_to_eye(self.parts, C, gaze_vector)

        # Update actors with transformed meshes
        for key in ["globe", "cornea", "iris", "pupil"]:
            self.actors[key].mapper.SetInputData(self.parts[key])

        self.plotter.render()

    def get_img(self):
        """Return a NumPy image (H x W x 3) of the current scene."""
        return self.plotter.screenshot(return_img=True)

    def close(self):
        self.plotter.close()

#
# if __name__ == "__main__":
#     demo_render_with_pose()

if __name__ == "__main__":
    renderer = EyeSceneRenderer()

    # Example loop: move eye around
    for x in np.linspace(-2, 2, 5):
        C = np.array([x, 0.0, 0.0])
        gaze = np.array([0.0, 0.0, 1.0])  # straight ahead
        renderer.set_eye_pose(C, gaze)
        img = renderer.get_img()


        plt.imshow(img); plt.show()

    renderer.close()
