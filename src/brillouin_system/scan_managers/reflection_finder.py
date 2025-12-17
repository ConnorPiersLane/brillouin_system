from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class ReflectionFindingResult:
    found: bool
    z_um: float | None
    metric: float | None
    threshold: float
    n_steps: int



class ReflectionFinder:
    def __init__(
        self,
        camera,
        zaber_axis,
        exposure_time_for_reflection_finding: float = 0.05,
        gain_for_reflection_finding: int = 0,
        reflection_threshold_value: float = 5000.0,
        step_distance_um_for_reflection_finding: int = 20,
        max_search_distance_um_for_reflection_finding: int = 2000,
        n_bg_images_for_reflection_finding: int = 10,
    ):
        self.camera = camera
        self.zaber_axis = zaber_axis

        self.exposure_time = exposure_time_for_reflection_finding
        self.gain = gain_for_reflection_finding
        self.threshold_value = reflection_threshold_value
        self.step_um = step_distance_um_for_reflection_finding
        self.max_distance_um = max_search_distance_um_for_reflection_finding
        self.n_bg_images = n_bg_images_for_reflection_finding

        self._bg_value: float | None = None
        self._bg_std: float | None = None

    # --- camera helpers ---
    def get_frame_value(self) -> float:
        frame, _ = self.camera.snap()
        # NOTE: sum() is okay but very sensitive to background/ROI.
        # Often better: frame.max() or np.percentile(frame, 99.9)
        return float(frame.sum())

    def take_and_store_background_value(self) -> None:
        vals = [self.get_frame_value() for _ in range(self.n_bg_images)]
        self._bg_value, self._bg_std = float(np.mean(vals)), float(np.std(vals))

    def get_reflection_metric(self, n_avg: int = 1) -> float:
        if self._bg_value is None:
            raise RuntimeError("Background not set. Call take_and_store_background_value() first.")
        vals = [self.get_frame_value() for _ in range(max(1, n_avg))]
        return float(np.mean(vals) - self._bg_value)

    # --- main algorithm ---
    def find_reflection(
        self,
        direction: int = +1,
        confirm_hits: int = 2,
        n_avg_per_step: int = 1,
        refine: bool = True,
        refine_half_window_um: int | None = None,
        refine_step_um: int | None = None,
        stop_after_peak_fall: bool = True,
        peak_fall_ratio: float = 0.7,
    ) -> ReflectionFindingResult:
        """
        Efficient reflection finding:
          1) Coarse scan until threshold confirmed.
          2) (Optional) Bracket/peak-follow for a few steps.
          3) (Optional) Fine scan around best_z.

        - stop_after_peak_fall: once hit is found, stop when metric falls below
          peak_fall_ratio * best_metric (a cheap way to avoid overscanning).
        """
        direction = 1 if direction >= 0 else -1
        step_um = self.step_um * direction
        max_um = self.max_distance_um

        self.take_and_store_background_value()
        threshold = 10 * self._bg_std

        z0 = self.zaber_axis.get_position()
        best_z = z0
        best_metric = -np.inf

        n_steps = int(np.floor(max_um / abs(step_um))) if abs(step_um) > 0 else 0
        consecutive = 0
        found = False
        steps_taken = 0

        # --- Coarse scan ---
        for i in range(n_steps + 1):
            steps_taken = i
            metric = self.get_reflection_metric(n_avg=n_avg_per_step)
            z = self.zaber_axis.get_position()

            if metric > best_metric:
                best_metric = metric
                best_z = z

            if metric >= threshold:
                consecutive += 1
                if consecutive >= max(1, int(confirm_hits)):
                    found = True
            else:
                consecutive = 0

            # If found, optionally stop once we clearly passed the peak (saves time)
            if found and stop_after_peak_fall and best_metric > 0:
                if metric < peak_fall_ratio * best_metric:
                    break

            if i < n_steps:
                self.zaber_axis.move_rel(step_um)


        # If never found and requested, go back home.
        if not found:
            self.zaber_axis.move_abs(z0)
            return ReflectionFindingResult(
                found=False,
                z_um=None,
                metric=None,
                threshold=threshold,
                n_steps=steps_taken,
            )


        # --- Refinement around best_z (fine search) ---
        if refine:
            if refine_half_window_um is None:
                refine_half_window_um = max(5 * abs(self.step_um), 100)  # default window
            if refine_step_um is None:
                refine_step_um = max(1, abs(self.step_um) // 5)           # default finer step

            # Move to start of refine window
            z_center = best_z
            z_start = z_center - refine_half_window_um
            z_end = z_center + refine_half_window_um


            self.zaber_axis.move_abs(z_start)


            best_metric2 = 0
            best_z2 = self.zaber_axis.get_position()

            n_ref_steps = int(np.floor((z_end - z_start) / refine_step_um))
            for _ in range(n_ref_steps + 1):
                metric = self.get_reflection_metric(n_avg=n_avg_per_step)
                z = float(self.zaber_axis.get_position())
                if metric > best_metric2:
                    best_metric2 = metric
                    best_z2 = z

                self.zaber_axis.move_rel(refine_step_um)


            best_z, best_metric = best_z2, best_metric2

        return ReflectionFindingResult(
            found=True,
            z_um=best_z,
            metric=best_metric,
            threshold=threshold,
            n_steps=steps_taken,
        )
