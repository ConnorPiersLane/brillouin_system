from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Callable

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.logging_utils.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ReflectionFindingResult:
    found: bool
    z_um: float | None


class ReflectionFinderNI:
    """
    Slew Z while monitoring NI analog input; stop when reflection detected.

    DAQ must look like your NI6008:
      - context manager: daq.streaming()
      - read_value(timeout_s=...) -> float
      - optionally get_background_value(n_samples) -> (mean, std)
    """

    def __init__(self, daq, zaber_axis):
        self.daq = daq
        self.zaber_lens: ZaberEyeLens = zaber_axis

    def find_reflection_plane(
        self,
        *,
        n_sigma: int = 5,
        speed_um_s: float = 1000.0,
        max_search_distance_um: float = 2000.0,
        n_bg_samples: int = 500,
        n_hits: int = 2,                  # require N consecutive hits (spike/noise robustness)
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> ReflectionFindingResult:

        z0 = self.zaber_lens.get_position()
        z_hit = z0

        with self.daq.streaming():
            # --- background (for choosing threshold) ---
            xs = self.daq.read_block(int(n_bg_samples))
            bg_mean = sum(xs) / len(xs)
            bg_std = (sum((x - bg_mean) ** 2 for x in xs) / len(xs)) ** 0.5

            # For delta mode, threshold is in "volts above baseline"
            threshold = float(bg_mean + n_sigma * bg_std)

            max_search_time = max_search_distance_um / speed_um_s

            # --- start guarded slewing ---
            self.zaber_lens.start_slewing_guarded(
                speed_um_per_s=float(speed_um_s),
                max_distance_um=float(max_search_distance_um),
            )

            t_start = time.monotonic()
            hits = 0

            try:
                while True:
                    if cancel_cb and cancel_cb():
                        log.info(f"[ReflectionFinderNI] Cancelled at {self.zaber_lens.get_position()} um")
                        self.zaber_lens.move_abs(z0)
                        return ReflectionFindingResult(found=False, z_um=None)

                    # time-based bailout
                    if (time.monotonic() - t_start) >= max_search_time:
                        log.info(f"[ReflectionFinderNI] Timeout at {self.zaber_lens.get_position()} um")
                        self.zaber_lens.move_abs(z0)
                        return ReflectionFindingResult(found=False, z_um=None)

                    # --- detection (short slice so we can keep checking cancel/distance) ---

                    v = self.daq.read_value(timeout_s=0.01)
                    if v > threshold:
                        hits += 1
                        if hits == 1:
                            self.zaber_lens.stop_slewing()
                            z_hit = self.zaber_lens.get_position()
                    else:
                        hits = 0

                    if hits >= n_hits:
                        self.zaber_lens.stop_slewing()
                        return ReflectionFindingResult(
                            found=True,
                            z_um=z_hit,
                        )



            finally:
                self.zaber_lens.stop_slewing()
