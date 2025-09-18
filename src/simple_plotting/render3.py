import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def _world_to_display(plotter, xyz):
    ren = plotter.renderer
    ren.SetWorldPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
    ren.WorldToDisplay()
    return ren.GetDisplayPoint()  # (x, y, z) with origin at bottom-left


def rotation_matrix_from_gaze(gaze_vector, up_hint=np.array([0, 1, 0])):
    """Returns a 3x3 rotation matrix aligning +Z with gaze_vector."""
    forward = gaze_vector / np.linalg.norm(gaze_vector)
    right = np.cross(up_hint, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R_mat = np.eye(3)
    R_mat[:3, 0] = right
    R_mat[:3, 1] = up
    R_mat[:3, 2] = forward
    return R_mat

class EyeGazeRenderer:
    EYE_COLORS = {
        "brown":  (np.array([0.35, 0.20, 0.05]), np.array([0.15, 0.07, 0.02])),
        "hazel":  (np.array([0.45, 0.30, 0.10]), np.array([0.20, 0.15, 0.05])),
        "green":  (np.array([0.35, 0.55, 0.30]), np.array([0.15, 0.25, 0.10])),
        "blue":   (np.array([0.4, 0.7, 0.9]),   np.array([0.1, 0.3, 0.5])),
        "grey":   (np.array([0.6, 0.65, 0.7]),  np.array([0.3, 0.35, 0.4])),
        "amber":  (np.array([0.8, 0.55, 0.2]),  np.array([0.45, 0.25, 0.05])),
    }

    @staticmethod
    def make_colored_sphere(radius, eye_color="blue"):
        iris_inner, iris_outer = EyeGazeRenderer.EYE_COLORS.get(
            eye_color, EyeGazeRenderer.EYE_COLORS["blue"]
        )

        sphere = pv.Sphere(radius=radius, theta_resolution=100, phi_resolution=100)
        points = sphere.points
        r = np.linalg.norm(points, axis=1)
        phi = np.arccos(points[:, 2] / r)

        colors = np.zeros((points.shape[0], 3))
        iris_max = np.deg2rad(30)
        pupil_max = np.deg2rad(12)

        for i, angle in enumerate(phi):
            if angle > iris_max:  # sclera
                colors[i] = [0.93, 0.93, 0.95]
            elif angle > pupil_max:  # iris
                t = (angle - pupil_max) / (iris_max - pupil_max)
                colors[i] = (1 - t) * iris_inner + t * iris_outer
            else:  # pupil
                colors[i] = [0.0, 0.0, 0.0]

        sphere["colors"] = colors
        return sphere

    def __init__(self, radius=7.7, eye_color="blue"):
        self.radius = radius
        self.eye_color = eye_color
        self.sphere = self.make_colored_sphere(radius, eye_color=eye_color)
        self.plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
        self.plotter.add_mesh(self.sphere, scalars="colors", rgb=True, smooth_shading=True, lighting=True)

        # defaults: realistic human iris/pupil
        self.iris_angle = np.arcsin((12.0 / 2.0) / self.radius)  # ≈ 12 mm iris
        self.current_pupil_diameter_mm = 2.0
        self.set_pupil_diameter(self.current_pupil_diameter_mm)


    def mm_to_pixels(self):
        ren = self.plotter.renderer
        ren.SetWorldPoint(0, 0, 0, 1.0)
        ren.WorldToDisplay()
        x0, y0, _ = ren.GetDisplayPoint()

        ren.SetWorldPoint(1.0, 0, 0, 1.0)
        ren.WorldToDisplay()
        x1, y1, _ = ren.GetDisplayPoint()

        px_per_mm_x = x1 - x0
        ren.SetWorldPoint(0, 1.0, 0, 1.0)
        ren.WorldToDisplay()
        _, y2, _ = ren.GetDisplayPoint()
        px_per_mm_y = y2 - y0
        return px_per_mm_x, px_per_mm_y, (x0, y0)

    def _camera_basis(self):
        cam = self.plotter.camera
        pos = np.array(cam.position, dtype=float)
        foc = np.array(cam.focal_point, dtype=float)
        forward = _normalize(foc - pos)
        up_raw = np.array(getattr(cam, "up", (0, 1, 0)), dtype=float)
        try:
            if up_raw is None or np.linalg.norm(up_raw) == 0:
                up_raw = np.array(cam.view_up, dtype=float)
        except Exception:
            pass
        right = _normalize(np.cross(forward, up_raw))
        up = _normalize(np.cross(right, forward))
        return right, up, forward, pos, foc

    def enable_orthographic(self):
        """Switch camera to orthographic projection and frame the eye correctly."""
        self.plotter.enable_parallel_projection()

        cam = self.plotter.camera
        cam.focal_point = (0.0, 0.0, 0.0)
        cam.position = (0.0, 0.0, 100 * self.radius)

        # Correct way to set view-up vector
        cam.SetViewUp(0.0, 1.0, 0.0)

        # Parallel scale is half the height of the visible scene in world units
        cam.parallel_scale = 1.3 * self.radius

        self.plotter.render()

    def draw_cross_cam_mm(self, img, x_mm, y_mm, size_mm=1.0,
                          color=(255, 0, 0), thickness=2, z_bump_mm=0.0):
        right, up, forward, _, _ = self._camera_basis()
        C = np.array([0.0, 0.0, 0.0], dtype=float)
        P = C + x_mm * right + y_mm * up + z_bump_mm * (-forward)
        x_disp, y_disp, _ = _world_to_display(self.plotter, P)
        w, h = self.plotter.window_size
        x_px, y_px = int(round(x_disp)), int(round(h - y_disp))

        Pref = P + size_mm * right
        x_ref, _, _ = _world_to_display(self.plotter, Pref)
        cross_len_px = max(1, int(round(abs(x_ref - x_disp))))

        x1, x2 = max(0, x_px - cross_len_px), min(w, x_px + cross_len_px)
        y1, y2 = max(0, y_px - cross_len_px), min(h, y_px + cross_len_px)

        img[max(0, y_px - thickness):min(h, y_px + thickness), x1:x2] = color
        img[y1:y2, max(0, x_px - thickness):min(w, x_px + thickness)] = color
        return img

    def set_iris_diameter(self, iris_diameter_mm):
        """Set iris diameter in mm and update mesh colors using current pupil size."""
        self.iris_angle = np.arcsin(min(1.0, (iris_diameter_mm / 2.0) / self.radius))
        # Re-apply pupil so colors are refreshed
        self.set_pupil_diameter(self.current_pupil_diameter_mm)

    def set_pupil_diameter(self, pupil_diameter_mm):
        """Set pupil diameter in mm and update mesh colors using current iris size."""
        self.current_pupil_diameter_mm = pupil_diameter_mm
        pupil_angle = np.arcsin(min(1.0, (pupil_diameter_mm / 2.0) / self.radius))
        # Keep pupil inside iris
        pupil_angle = min(pupil_angle, self.iris_angle * 0.98)

        iris_inner, iris_outer = self.EYE_COLORS[self.eye_color]
        points = self.sphere.points
        r = np.linalg.norm(points, axis=1)
        phi = np.arccos(points[:, 2] / r)

        colors = np.zeros((points.shape[0], 3))
        for i, angle in enumerate(phi):
            if angle > self.iris_angle:  # sclera
                colors[i] = [0.93, 0.93, 0.95]
            elif angle > pupil_angle:  # iris gradient
                t = (angle - pupil_angle) / (self.iris_angle - pupil_angle + 1e-12)
                colors[i] = (1 - t) * iris_inner + t * iris_outer
            else:  # pupil
                colors[i] = [0.0, 0.0, 0.0]

        self.sphere["colors"] = colors
        if self.plotter.ren_win:
            self.plotter.render()

    def set_eye_pose(self, C, gaze_vector):
        """Move and rotate eye so that corneal center = C, gaze = gaze_vector."""
        R_mat = rotation_matrix_from_gaze(gaze_vector)
        # Apply transform to the mesh
        transform = np.eye(4)
        transform[:3, :3] = R_mat
        transform[:3, 3] = C
        self.sphere.transform(transform, inplace=True)

        if self.plotter.ren_win:
            self.plotter.render()

    def render_eye(self, yaw_deg=40, pitch_deg=0):
        yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
        direction = np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])
        distance = 4 * self.radius
        camera_pos = distance * direction
        self.plotter.camera_position = [tuple(camera_pos), (0, 0, 0), (0, 1, 0)]
        img = self.plotter.screenshot(return_img=True)


        img = self.draw_cross_cam_mm(img, 0, 0)
        return img

    def close(self):
        self.plotter.close()


if __name__ == "__main__":
    renderer = EyeGazeRenderer(radius=7.7)
    renderer.enable_orthographic()

    # Example: corneal center 2 mm right, gaze slightly nasal
    C = np.array([2.0, 1.0, -10000.0])
    gaze_vector = _normalize([0.1, 0.3, 1.0])  # slightly off-axis
    renderer.set_eye_pose(C, gaze_vector)

    img = renderer.render_eye(yaw_deg=0, pitch_deg=0)  # no camera rotation
    renderer.close()

    plt.imshow(img)
    plt.axis("off")
    plt.show()
