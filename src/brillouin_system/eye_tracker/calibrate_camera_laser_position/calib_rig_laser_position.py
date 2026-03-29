import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import os
import tomli
import tomli_w
import json
from dataclasses import asdict

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterface
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.my_dataclasses.my_exceptions import OperationCancelled
from brillouin_system.scan_managers.ni_reflection_finder4 import find_reflection_realtime, ReflectionResult
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig

log = get_logger(__name__)
SETTINGS_FILE_TOML_PATH = Path(__file__).parent.resolve() / "settings.toml"
FILENAME = "offset.toml"

@dataclass
class MeasuredCircle:
    camera_estimates_of_pupil_center_xxyyzz: list[tuple[float, float, float]]
    reflection_boundary_points_xxyyzz: list[tuple[float, float, float]]
    center_fitted_from_reflection_boundary_xxyyzz: tuple[float, float, float]
    radius_fitted_from_reflection_boundary: float
    laser_offset_dxdydz: tuple[float, float, float]



def save_measured_circle_to_json(measured: MeasuredCircle, filename: str = "measured_circle.json"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename)

    # Convert dataclass → dict
    data = asdict(measured)

    # Ensure tuples → lists (JSON-safe)
    def convert(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    data = convert(data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return filepath

def load_calibration_settings() -> dict:
    if not SETTINGS_FILE_TOML_PATH.exists():
        raise FileNotFoundError(f"Settings file not found: {SETTINGS_FILE_TOML_PATH}")

    with open(SETTINGS_FILE_TOML_PATH, "rb") as f:
        data = tomli.load(f)

    if "calibration" not in data:
        raise ValueError("Missing [calibration] section in settings.toml")

    return data

class LaserOffset:
    def __init__(self, dx, dy, dz):
        """
        [um]: the position of the laser in zaber or rig coordinates
        Args:
            dx: [um] from rig to laser
            dy: "
            dz: "
        """
        self.dx = dx
        self.dy = dy
        self.dz = dz



def load_laser_coord_system_from_toml() -> LaserOffset:
    """
    Load LaserCoordSystem from a TOML file in the same directory as this file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, FILENAME)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"TOML file not found: {filepath}")

    with open(filepath, "rb") as f:  # ⚠️ binary mode required
        data = tomli.load(f)

    if "offset" not in data:
        raise ValueError("Invalid TOML: missing 'offset' section")

    offset = data["offset"]

    dx = float(offset.get("dx", 0.0))
    dy = float(offset.get("dy", 0.0))
    dz = float(offset.get("dz", 0.0))

    return LaserOffset(dx=dx, dy=dy, dz=dz)


def save_offset_to_toml(dx: float, dy: float, dz: float, filename: str = "offset.toml"):
    """
    Save dx, dy, dz (dz=0) to a TOML file in the same directory as this file.
    """

    data = {
        "offset": {
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
        }
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename)

    with open(filepath, "wb") as f:  # ⚠️ binary mode required
        tomli_w.dump(data, f)

    return filepath

class CalibRigLaserPosition:
    """
    Two coordinate systems are used here:
    x, y, z: ‘Rig’ or ‘Zaber’ or ‘Ref’ coordinate system. 0,0,0 moves with the zaber axis
    xx, yy, zz: absolute position of the zaber axis. defined by zaber.get_position()
                fixed, does not move with zaber axis, is absolute reference
                xx, yy, zz from zaber human interface
    By design x,y,z are parallel to xx,yy,zz

    Important, it is assumed that the z-coordinate origin is at zaberlens.move_abs(0), at least approximately.

    """

    def __init__(
        self,
        ni,
        zaber_eye_lens,
        zaber_hi,
        get_pupil_center_ref: Callable[[], tuple[float, float, float]],
        cancel_callback: Callable,
        axial_scan_config: ScanningConfig,
    ):
        # Load scanning settings:
        settings = load_calibration_settings()
        cfg = settings["calibration"]
        self._dphi_deg = cfg.get("dphi_deg")
        self._coarse_step_um = cfg.get("coarse_step_um")
        self._max_steps = cfg.get("max_steps")
        self._tolerance_um = cfg.get("tolerance_um")
        self._max_binary_iters = cfg.get("max_binary_iters")
        self._n_confirmations = cfg.get("n_confirmations")
        self._recenter_n_moves = cfg.get("recenter_n_moves")
        self._recenter_settle_s = cfg.get("recenter_settle_s")
        self._backstep_um = cfg.get("backstep_um")

        # Assign
        self.ni = ni
        self.zaber_eye_lens: ZaberEyeLens = zaber_eye_lens
        self.cancel_callback = cancel_callback

        self.zaber_hi: ZaberHumanInterface = zaber_hi
        self.get_pupil_center_ref: Callable[[], tuple[float, float, float]] = get_pupil_center_ref
        self._axial_scan_config: ScanningConfig = axial_scan_config

        self._init_zaber_hi_position = zaber_hi.get_position()
        self._zaber_lens_z0 = self.zaber_eye_lens.get_position()
        self._zaber_lens_search_position = self._zaber_lens_z0 - self._backstep_um
        self._camera_estimates_of_pupil_center_xxyyzz: list[tuple[float, float, float]] = []
        self._reflection_boundary_points_xxyyzz: list[tuple[float, float, float]] = []


    def init_position_to_estimated_pupil_center(self) -> tuple[float, float, float]:
        """
        Repeatedly use camera estimate to move toward the pupil center.
        Then store the resulting absolute xxyyzz position.
        """
        self.zaber_hi.move_abs(*self._init_zaber_hi_position)
        for _ in range(self._recenter_n_moves):
            if self.cancel_callback():
                log.info(f"Cancelled.")
                raise OperationCancelled()

            time.sleep(self._recenter_settle_s)
            center = self.get_pupil_center_ref()
            print(f"Pupil center calib: {center}")
            # convert from mm to um
            x, y, z = center[0]*1000, center[1]*1000, center[2]*1000

            # Assumint Rig COS and zaber_lens share same origin
            # Moving the Rig here (not the lens)
            self.zaber_hi.move_rel(dx=x-0, dy=y-0, dz=z-self._zaber_lens_z0)

        return self.zaber_hi.get_position()


    def move_dr_at_constant_angle_phi(self, phi_deg: float, dr: float):
        phi_rad = math.radians(phi_deg)
        dx = math.cos(phi_rad) * dr
        dy = math.sin(phi_rad) * dr
        self.zaber_hi.move_rel(dx=dx, dy=dy, dz=None)

    def move_to_xxyy(self, xx: float, yy: float):
        self.zaber_hi.move_abs(x=xx, y=yy, z=None)


    def _interp_xy(
        self,
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
    ) -> tuple[float, float]:
        x = p0[0] + 0.5 * (p1[0] - p0[0])
        y = p0[1] + 0.5 * (p1[1] - p0[1])
        return x, y

    def _xy_distance(
        self,
        p0: tuple[float, float, float],
        p1: tuple[float, float, float]
    ) -> float:
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    def find_reflection_plane(self):

        # Move the zaber lens back:
        self.zaber_eye_lens.move_abs(self._zaber_lens_search_position)

        result: ReflectionResult = find_reflection_realtime(
            ni=self.ni,
            zaber=self.zaber_eye_lens,
            ni_sample_rate_hz=self._axial_scan_config.ni_sample_rate_hz,
            speed_um_s=self._axial_scan_config.speed_um_s,
            max_distance_um=self._axial_scan_config.max_distance_um,
            threshold_high_n_sigma=self._axial_scan_config.threshold_high_n_sigma,
            threshold_low_n_sigma=self._axial_scan_config.threshold_low_n_sigma,
            bg_acqui_s=self._axial_scan_config.bg_acqui_s,
            debounce_s=self._axial_scan_config.debounce_s,
            z_poll_s=self._axial_scan_config.z_poll_s,
            chunk_size=self._axial_scan_config.chunk_size,
            idle_sleep_s=self._axial_scan_config.idle_sleep_s,
            z_offset_um=self._axial_scan_config.z_offset_um,
        )

        return result

    def is_found_reflection_plane(self) -> tuple[bool, tuple[float, float, float] | None]:
        result = self.find_reflection_plane()
        if result.found:
            z0 = self._zaber_lens_z0 # this is where the plane should be
            z_is = result.event_z_um # this is where the plane is found
            zaber_hi_pos = self.zaber_hi.get_position()
            pos_plane = (
                zaber_hi_pos[0],
                zaber_hi_pos[1],
                zaber_hi_pos[2] + (z_is - z0),
            )
        else:
            pos_plane = None
        return result.found, pos_plane


    def move_out_until_reflection_plane(
        self,
        angle_deg: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        Move outward along a ray until the reflection plane is found.

        Returns:
            (last_not_found_position, first_found_position)
        """
        start_pos = self.zaber_hi.get_position()

        if self.is_found_reflection_plane()[0]:
            raise ValueError(
                f"Starting position at angle {angle_deg} deg already appears to be inside the reflection region."
            )

        last_not_found_position = start_pos

        for _ in range(self._max_steps):
            if self.cancel_callback():
                log.info(f"Cancelled.")
                raise OperationCancelled()
            self.move_dr_at_constant_angle_phi(phi_deg=angle_deg, dr=self._coarse_step_um)
            current_pos = self.zaber_hi.get_position()

            is_found, pos = self.is_found_reflection_plane()
            if is_found:
                first_found_position = pos
                return last_not_found_position, first_found_position

            last_not_found_position = current_pos

        raise RuntimeError(
            f"Reflection plane was not found within max_steps={self._max_steps} at angle {angle_deg} deg."
        )

    def binary_search_to_refine(
        self,
        not_found_pos: tuple[float, float, float],
        found_pos: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """
        Refine the reflection boundary between:
          - not_found_pos: a position where reflection is NOT found
          - found_pos: a position where reflection IS found

        Returns:
            Estimated boundary position (xxyyzz).
        """
        lo = not_found_pos
        hi = found_pos

        self.move_to_xxyy(lo[0], lo[1])
        self.move_to_xxyy(hi[0], hi[1])


        for _ in range(self._max_binary_iters):
            if self.cancel_callback():
                log.info(f"Cancelled.")
                raise OperationCancelled()
            if self._xy_distance(lo, hi) <= self._tolerance_um:
                break

            mid_x, mid_y = self._interp_xy(lo, hi)
            self.move_to_xxyy(mid_x, mid_y)
            mid_pos = (mid_x, mid_y, hi[2])

            is_found, pos = self.is_found_reflection_plane()
            if is_found:
                hi = pos
            else:
                lo = mid_pos

        self.move_to_xxyy(hi[0], hi[1])
        return hi

    def scan_boundary_point_at_angle(self, angle_deg: float) -> tuple[float, float, float]:
        """
        1. Recenter to camera-estimated pupil center
        2. Store that center xxyyzz
        3. Search outward along the given angle until the boundary is bracketed
        4. Refine boundary with binary search
        5. Store and return the refined boundary point
        """


        last_not_found, first_found = self.move_out_until_reflection_plane(
            angle_deg=angle_deg,
        )

        boundary = self.binary_search_to_refine(
            not_found_pos=last_not_found,
            found_pos=first_found,
        )

        return boundary

    def scan_boundary_over_angles(self) -> None:
        """
        Recenter from cameras and scan one boundary point for each angle:
        0, dphi, 2*dphi, ... up to < 360 (or <= 360 if include_endpoint_360=True)
        """
        dphi_deg = self._dphi_deg
        self._camera_estimates_of_pupil_center_xxyyzz = [] # reset
        self._reflection_boundary_points_xxyyzz = []


        n_steps = int(math.floor(360.0 / dphi_deg))
        for i in range(n_steps):
            angle_deg = i * dphi_deg
            # init
            center = self.init_position_to_estimated_pupil_center()
            self._camera_estimates_of_pupil_center_xxyyzz.append(center)

            boundary = self.scan_boundary_point_at_angle(angle_deg=angle_deg)
            self._reflection_boundary_points_xxyyzz.append(boundary)


    def fit_circle_3d(self) -> tuple[float, float, float, float]:
        """
        Fit a circle in 3D:
        1. Fit a best-fit plane to the 3D points
        2. Project points into that plane
        3. Fit a 2D circle in plane coordinates
        4. Transform fitted center back to 3D

        Returns:
            (cx, cy, cz, radius)
        """
        points = self._reflection_boundary_points_xxyyzz
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a 3D circle")

        pts = np.array(points, dtype=float)  # shape (N, 3)

        # ---- 1) Fit plane with PCA/SVD ----
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid

        # Vt rows are principal directions; last row is plane normal
        _, _, vt = np.linalg.svd(pts_centered, full_matrices=False)

        # Plane basis vectors (orthonormal, spanning the plane)
        e1 = vt[0, :]
        e2 = vt[1, :]

        # ---- 2) Project 3D points into 2D plane coordinates ----
        xs_2d = pts_centered @ e1
        ys_2d = pts_centered @ e2

        # ---- 3) Fit 2D circle in plane coordinates ----
        A_mat = np.column_stack((xs_2d, ys_2d, np.ones_like(xs_2d)))
        b_vec = -(xs_2d ** 2 + ys_2d ** 2)

        params, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        A, B, C = params

        cx_2d = -A / 2.0
        cy_2d = -B / 2.0

        radius_sq = cx_2d ** 2 + cy_2d ** 2 - C
        if radius_sq < 0:
            if radius_sq > -1e-9:
                radius_sq = 0.0
            else:
                raise RuntimeError(f"Negative radius_sq={radius_sq}")

        radius = math.sqrt(radius_sq)

        # ---- 4) Transform fitted center back to 3D ----
        center_3d = centroid + cx_2d * e1 + cy_2d * e2
        cx, cy, cz = center_3d.tolist()

        return cx, cy, cz, radius

    def run_calibration(self):
        self.scan_boundary_over_angles()
        cx, cy, cz, r = self.fit_circle_3d()

        # Mean camera estimate
        xs = [p[0] for p in self._camera_estimates_of_pupil_center_xxyyzz]
        ys = [p[1] for p in self._camera_estimates_of_pupil_center_xxyyzz]
        zs = [p[2] for p in self._camera_estimates_of_pupil_center_xxyyzz]

        cam_x = sum(xs) / len(xs)
        cam_y = sum(ys) / len(ys)
        cam_z = sum(zs) / len(zs)

        dx = cx - cam_x
        dy = cy - cam_y
        dz = cz - cam_z

        # Save offset (existing)
        save_offset_to_toml(dx=dx, dy=dy, dz=dz, filename=FILENAME)

        # ---- NEW: create MeasuredCircle ----
        measured = MeasuredCircle(
            camera_estimates_of_pupil_center_xxyyzz=self._camera_estimates_of_pupil_center_xxyyzz,
            reflection_boundary_points_xxyyzz=self._reflection_boundary_points_xxyyzz,
            center_fitted_from_reflection_boundary_xxyyzz=(cx, cy, cz),
            radius_fitted_from_reflection_boundary=r,
            laser_offset_dxdydz=(dx, dy, dz),
        )

        save_measured_circle_to_json(measured)

        print(f"Saved calibration to {FILENAME}")
        print(f"Saved measured circle to measured_circle.json")
        print(f"Center: ({cx:.3f}, {cy:.3f}, {cz:.3f}), Radius: {r:.3f}")

        return LaserOffset(dx=dx, dy=dy, dz=dz)