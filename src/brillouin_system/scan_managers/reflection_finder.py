from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time

import numpy as np

from brillouin_system.devices.cameras.andor.baseCamera import BaseCamera
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens


@dataclass
class ReflectionFindingResult:
    found: bool
    z_um: float | None



ANDROR_CAM_SETTINGS ={
    "flip_image_horizontally": False,
    "vss_speed": 0,
    "fixed_pre_amp_mode_index": 0,
}




class ReflectionFinder:
    def __init__(
        self,
        camera,
        zaber_axis,
    ):
        self.camera: BaseCamera = camera
        self.zaber_lens: ZaberEyeLens = zaber_axis

    @contextmanager
    def camera_fast_emccd_mode(self, exposure_time, emccd_gain):
        _old_camera_info = self.camera.get_camera_info_dataclass()

        # Change Settings
        self.camera.set_fixed_pre_amp_mode(index=ANDROR_CAM_SETTINGS["fixed_pre_amp_mode_index"])
        self.camera.set_vss_index(index=ANDROR_CAM_SETTINGS["vss_speed"])
        self.camera.set_flip_image_horizontally(flip=ANDROR_CAM_SETTINGS["flip_image_horizontally"])
        self.camera.set_exposure_time(exposure_time)
        self.camera.set_emccd_gain(emccd_gain)

        try:
            yield
        finally:
            self.camera.set_fixed_pre_amp_mode(index=_old_camera_info.fixed_pre_amp_mode_index)
            self.camera.set_vss_index(index=_old_camera_info.vss_speed)
            self.camera.set_flip_image_horizontally(flip=_old_camera_info.flip_image_horizontally)
            self.camera.set_exposure_time(_old_camera_info.exposure)
            self.camera.set_emccd_gain(_old_camera_info.gain)

    def get_frame_value(self) -> float:
        """Wait for a *new* streaming frame up to `timeout` seconds; return sum or None."""
        timeout = 2
        t0 = time.monotonic()
        while True:
            frame = self.camera.get_newest_streaming_image()
            if frame is not None:
                return float(frame.sum())

            if time.monotonic() - t0 >= timeout:
                raise RuntimeError("No streaming Frame arrived in time")

            time.sleep(0.001)

    def get_background_value(self, n_bg_images) -> tuple[float, float]:
        vals = [self.get_frame_value() for _ in range(n_bg_images)]
        _bg_mean, _bg_std = float(np.mean(vals)), float(np.std(vals))
        return _bg_mean, _bg_std


    def find_reflection_plane(
            self,
            exposure_time: float,
            gain: int,
            n_sigma: float,
            speed_um_s: float = 1000,
            max_search_distance_um: float = 2000,
            n_bg_images: int = 10,
    ) -> ReflectionFindingResult:
        """
        Slew Z while monitoring streaming images until reflection detected.

        Exit conditions:
          - reflection found (threshold crossing)
          - travelled distance >= max_search_distance_um_for_reflection_finding
          - optional timeout_s
        """

        with self.camera_fast_emccd_mode(exposure_time=exposure_time, emccd_gain=gain):
            with self.camera.streaming():
                # --- Background estimate (robust to None frames) ---

                bg_mean, bg_std = self.get_background_value(n_bg_images)
                threshold = bg_mean + n_sigma * bg_std

                # --- Start guarded slewing ---
                z0 = float(self.zaber_lens.get_position())

                self.zaber_lens.start_slewing(speed_um_per_s=speed_um_s)

                try:
                    while True:
                        # --- Stop condition 2: travelled distance (guard likely stopped already) ---
                        z = float(self.zaber_lens.get_position())
                        if abs(z - z0) >= abs(max_search_distance_um):
                            self.zaber_lens.stop_slewing()
                            return ReflectionFindingResult(found=False, z_um=None)
                        # --- Read next frame (may be None) ---
                        frame_value = self.get_frame_value()

                        # --- Detection ---
                        if frame_value > threshold:
                            self.zaber_lens.stop_slewing()
                            return ReflectionFindingResult(found=True, z_um=z)


                finally:
                    # Always stop slewing when leaving the function
                    self.zaber_lens.stop_slewing()
