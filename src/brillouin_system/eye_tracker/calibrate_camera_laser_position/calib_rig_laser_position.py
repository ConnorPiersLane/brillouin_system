import math
import time
from typing import Callable

import numpy as np

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterface
from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults
from brillouin_system.scan_managers.ni_reflection_finder4 import find_reflection_realtime, ReflectionResult
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig


class CalibRigLaserPosition:
    """
    Two coordinate systems are used here:
    x, y, z: ‘Rig’ or ‘Zaber’ or ‘Ref’ coordinate system. 0,0,0 moves with the zaber axis
    xx, yy, zz: absolute position of the zaber axis. defined by zaber.get_position()
                fixed, does not move with zaber axis, is absolute reference
                xx, yy, zz from zaber human interface
    By design x,y,z are parallel to xx,yy,zz
    """

    def __init__(
        self,
        ni,
        zaber_eye_lens,
        zaber_hi,
        get_eyetracker_results: Callable[[], EyeTrackerResults],
        axial_scan_config: ScanningConfig
    ):
        self.ni = ni
        self.zaber_eye_lens: ZaberEyeLens = zaber_eye_lens
        self.zaber_hi: ZaberHumanInterface = zaber_hi
        self.get_eyetracker_results: Callable[[], EyeTrackerResults] = get_eyetracker_results
        self._axial_scan_config: ScanningConfig = axial_scan_config

        self._camera_estimates_of_pupil_center_xxyyzz: list[tuple[float, float, float]] = []
        self._reflection_boundary_points_xxyyzz: list[tuple[float, float, float]] = []
        self._pupil_center_fitted_from_reflection_boundary_xxyyzz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def initialize_position(self):
        time.sleep(1)

    def move_to_pupil_center_estimated_by_cameras(self):
        x, y = self.get_pupil_center_xy_from_cameras()
        self.move_dxdy_zaber_axis(dx=x, dy=y)

    def init_position_to_estimated_pupil_center(self, n_moves: int = 3, settle_s: float = 1.0):
        """
        Repeatedly use camera estimate to move toward the pupil center.
        Then store the resulting absolute xxyyzz position.
        """
        for _ in range(n_moves):
            time.sleep(settle_s)
            self.move_to_pupil_center_estimated_by_cameras()

        center_xxyyzz = self.zaber_hi.get_position()
        self._camera_estimates_of_pupil_center_xxyyzz.append(center_xxyyzz)
        return center_xxyyzz

    def get_pupil_center_xy_from_cameras(self) -> tuple[float, float]:
        time.sleep(0.5)  # some time rest to get actual prediction
        et_result: EyeTrackerResults = self.get_eyetracker_results()
        center = et_result.pupil3d.center_ref
        x, y, _z = center[0], center[1], center[2]
        return x, y

    def move_dxdy_zaber_axis(self, dx, dy):
        self.zaber_hi.move_rel(dx=dx, dy=dy, dz=None)

    def move_dr_at_constant_angle_phi(self, phi_deg: float, dr: float):
        phi_rad = math.radians(phi_deg)
        dx = math.cos(phi_rad) * dr
        dy = math.sin(phi_rad) * dr
        self.move_dxdy_zaber_axis(dx=dx, dy=dy)

    def move_to_xxyy(self, xx: float, yy: float):
        self.zaber_hi.move_abs(x=xx, y=yy, z=None)

    def move_to_xxyyzz(self, xx: float, yy: float, zz: float | None = None):
        self.zaber_hi.move_abs(x=xx, y=yy, z=zz)

    def _interp_xy(
        self,
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        t: float
    ) -> tuple[float, float]:
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        return x, y

    def _xy_distance(
        self,
        p0: tuple[float, float, float],
        p1: tuple[float, float, float]
    ) -> float:
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    def find_reflection_plane(self):
        ni_sample_rate_hz = self._axial_scan_config.ni_sample_rate_hz
        speed_um_s = self._axial_scan_config.speed_um_s
        max_distance_um = self._axial_scan_config.max_distance_um
        threshold_high_n_sigma = self._axial_scan_config.threshold_high_n_sigma
        threshold_low_n_sigma = self._axial_scan_config.threshold_low_n_sigma
        bg_acqui_s = self._axial_scan_config.bg_acqui_s
        debounce_s = self._axial_scan_config.debounce_s
        z_poll_s = self._axial_scan_config.z_poll_s
        chunk_size = self._axial_scan_config.chunk_size
        idle_sleep_s = self._axial_scan_config.idle_sleep_s
        offset_z_um = self._axial_scan_config.z_offset_um

        result: ReflectionResult = find_reflection_realtime(
            ni=self.ni,
            zaber=self.zaber_eye_lens,
            ni_sample_rate_hz=ni_sample_rate_hz,
            speed_um_s=speed_um_s,
            max_distance_um=max_distance_um,
            threshold_high_n_sigma=threshold_high_n_sigma,
            threshold_low_n_sigma=threshold_low_n_sigma,
            bg_acqui_s=bg_acqui_s,
            debounce_s=debounce_s,
            z_poll_s=z_poll_s,
            chunk_size=chunk_size,
            idle_sleep_s=idle_sleep_s,
            z_offset_um=offset_z_um,
        )
        return result

    def is_found_reflection_plane(self):
        result = self.find_reflection_plane()
        return result.found

    def robust_is_found_reflection_plane(self, n_confirmations: int = 3) -> bool:
        votes = 0
        for _ in range(n_confirmations):
            if self.is_found_reflection_plane():
                votes += 1
        return votes >= (n_confirmations // 2 + 1)

    def move_out_until_reflection_plane(
        self,
        angle_deg: float,
        step_size_um: float,
        max_steps: int = 100,
        n_confirmations: int = 3
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        Move outward along a ray until the reflection plane is found.

        Returns:
            (last_not_found_position, first_found_position)
        """
        start_pos = self.zaber_hi.get_position()

        if self.robust_is_found_reflection_plane(n_confirmations=n_confirmations):
            raise ValueError(
                f"Starting position at angle {angle_deg} deg already appears to be inside the reflection region."
            )

        last_not_found_position = start_pos

        for _ in range(max_steps):
            self.move_dr_at_constant_angle_phi(phi_deg=angle_deg, dr=step_size_um)
            current_pos = self.zaber_hi.get_position()

            if self.robust_is_found_reflection_plane(n_confirmations=n_confirmations):
                first_found_position = current_pos
                return last_not_found_position, first_found_position

            last_not_found_position = current_pos

        raise RuntimeError(
            f"Reflection plane was not found within max_steps={max_steps} at angle {angle_deg} deg."
        )

    def binary_search_to_refine(
        self,
        not_found_pos: tuple[float, float, float],
        found_pos: tuple[float, float, float],
        tolerance_um: float = 5.0,
        max_iters: int = 20,
        n_confirmations: int = 3,
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
        if self.robust_is_found_reflection_plane(n_confirmations=n_confirmations):
            raise ValueError("not_found_pos appears to already be inside the reflection region.")

        self.move_to_xxyy(hi[0], hi[1])
        if not self.robust_is_found_reflection_plane(n_confirmations=n_confirmations):
            raise ValueError("found_pos does not appear to be inside the reflection region.")

        for _ in range(max_iters):
            if self._xy_distance(lo, hi) <= tolerance_um:
                break

            mid_x, mid_y = self._interp_xy(lo, hi, 0.5)
            self.move_to_xxyy(mid_x, mid_y)
            mid_pos = self.zaber_hi.get_position()

            if self.robust_is_found_reflection_plane(n_confirmations=n_confirmations):
                hi = mid_pos
            else:
                lo = mid_pos

        self.move_to_xxyy(hi[0], hi[1])
        return self.zaber_hi.get_position()

    def scan_boundary_point_at_angle(
        self,
        angle_deg: float,
        coarse_step_um: float,
        max_steps: int = 100,
        tolerance_um: float = 5.0,
        max_binary_iters: int = 20,
        n_confirmations: int = 3,
        recenter_n_moves: int = 3,
        recenter_settle_s: float = 1.0,
    ) -> tuple[float, float, float]:
        """
        1. Recenter to camera-estimated pupil center
        2. Store that center xxyyzz
        3. Search outward along the given angle until the boundary is bracketed
        4. Refine boundary with binary search
        5. Store and return the refined boundary point
        """
        self.init_position_to_estimated_pupil_center(
            n_moves=recenter_n_moves,
            settle_s=recenter_settle_s,
        )

        last_not_found, first_found = self.move_out_until_reflection_plane(
            angle_deg=angle_deg,
            step_size_um=coarse_step_um,
            max_steps=max_steps,
            n_confirmations=n_confirmations,
        )

        boundary = self.binary_search_to_refine(
            not_found_pos=last_not_found,
            found_pos=first_found,
            tolerance_um=tolerance_um,
            max_iters=max_binary_iters,
            n_confirmations=n_confirmations,
        )

        self._reflection_boundary_points_xxyyzz.append(boundary)
        return boundary

    def scan_boundary_over_angles(
        self,
        dphi_deg: float,
        coarse_step_um: float,
        max_steps: int = 100,
        tolerance_um: float = 5.0,
        max_binary_iters: int = 20,
        n_confirmations: int = 3,
        recenter_n_moves: int = 3,
        recenter_settle_s: float = 1.0,
        include_endpoint_360: bool = False,
        continue_on_error: bool = False,
    ) -> list[tuple[float, float, float]]:
        """
        Recenter from cameras and scan one boundary point for each angle:
        0, dphi, 2*dphi, ... up to < 360 (or <= 360 if include_endpoint_360=True)
        """
        if dphi_deg <= 0:
            raise ValueError("dphi_deg must be > 0")

        self._reflection_boundary_points_xxyyzz = []

        angle_deg = 0.0
        end_deg = 360.0 if include_endpoint_360 else (360.0 - 1e-9)

        while angle_deg <= end_deg:
            try:
                self.scan_boundary_point_at_angle(
                    angle_deg=angle_deg,
                    coarse_step_um=coarse_step_um,
                    max_steps=max_steps,
                    tolerance_um=tolerance_um,
                    max_binary_iters=max_binary_iters,
                    n_confirmations=n_confirmations,
                    recenter_n_moves=recenter_n_moves,
                    recenter_settle_s=recenter_settle_s,
                )
            except Exception:
                if not continue_on_error:
                    raise
            angle_deg += dphi_deg

        return list(self._reflection_boundary_points_xxyyzz)

    def fit_circle_to_boundary_points(self):
        cx, cy, r = self.fit_circle_xy()

        self._pupil_center_fitted_from_reflection_boundary_xxyyzz = (cx, cy, 0.0)

        return (cx, cy), r

    def fit_circle_xy(
            self,
            boundary_points_xxyyzz: list[tuple[float, float, float]] | None = None
    ) -> tuple[float, float, float]:
        """
        Fit a circle in XY using least squares.

        Returns:
            (cx, cy, radius)
        """
        points = boundary_points_xxyyzz or self._reflection_boundary_points_xxyyzz

        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")

        # Extract XY
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        # Build linear system:
        # x^2 + y^2 + A*x + B*y + C = 0
        # -> A*x + B*y + C = -(x^2 + y^2)
        A_mat = np.column_stack((xs, ys, np.ones_like(xs)))
        b_vec = -(xs ** 2 + ys ** 2)

        # Solve least squares
        params, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        A, B, C = params

        # Convert to center + radius
        cx = -A / 2.0
        cy = -B / 2.0
        radius = math.sqrt(cx ** 2 + cy ** 2 - C)

        return cx, cy, radius


    def estimate_true_center_from_reflection_boundaries(
        self,
        dphi_deg: float,
        coarse_step_um: float,
        max_steps: int = 100,
        tolerance_um: float = 5.0,
        max_binary_iters: int = 20,
        n_confirmations: int = 3,
        recenter_n_moves: int = 3,
        recenter_settle_s: float = 1.0,
        include_endpoint_360: bool = False,
        continue_on_error: bool = False,
    ) -> tuple[tuple[float, float], float]:
        """
        Full pipeline:
        - For many angles:
            - move to camera-estimated center
            - store center xxyyzz
            - find radial boundary point
        - Fit circle through all boundary points
        - Return fitted true center xxyyzz and radius
        """
        self.scan_boundary_over_angles(
            dphi_deg=dphi_deg,
            coarse_step_um=coarse_step_um,
            max_steps=max_steps,
            tolerance_um=tolerance_um,
            max_binary_iters=max_binary_iters,
            n_confirmations=n_confirmations,
            recenter_n_moves=recenter_n_moves,
            recenter_settle_s=recenter_settle_s,
            include_endpoint_360=include_endpoint_360,
            continue_on_error=continue_on_error,
        )

        return self.fit_circle_to_boundary_points()

    def get_camera_estimates_of_pupil_center_xxyyzz(self) -> list[tuple[float, float, float]]:
        return list(self._camera_estimates_of_pupil_center_xxyyzz)

    def get_reflection_boundary_points_xxyyzz(self) -> list[tuple[float, float, float]]:
        return list(self._reflection_boundary_points_xxyyzz)

    def get_pupil_center_fitted_from_reflection_boundary_xxyyzz(self) -> tuple[float, float, float]:
        return self._pupil_center_fitted_from_reflection_boundary_xxyyzz

def compute_camera_to_fitted_center_offset(self) -> tuple[float, float]:
    """
    Compute dx, dy offset between:
    - mean camera-estimated pupil center
    - fitted center from reflection boundary

    Returns:
        (dx, dy) such that:
            corrected_position = camera_position + (dx, dy)
    """
    if not self._camera_estimates_of_pupil_center_xxyyzz:
        raise ValueError("No camera estimates available")

    if not hasattr(self, "_pupil_center_fitted_from_reflection_boundary_xxyy"):
        raise ValueError("Fitted center not available")

    # Mean camera estimate
    xs = [p[0] for p in self._camera_estimates_of_pupil_center_xxyyzz]
    ys = [p[1] for p in self._camera_estimates_of_pupil_center_xxyyzz]

    cam_x = sum(xs) / len(xs)
    cam_y = sum(ys) / len(ys)

    # Fitted center
    cx, cy = self._pupil_center_fitted_from_reflection_boundary_xxyy

    dx = cx - cam_x
    dy = cy - cam_y

    return dx, dy