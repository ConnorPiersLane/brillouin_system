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
            n_sigma: int = 6,
            speed_um_s: float = 1000.0,
            max_search_distance_um: float = 2000.0,
            n_bg_samples: int = 500,
            n_hits: int = 2,
            cancel_cb: Optional[Callable[[], bool]] = None,

            # new knobs:
            refine: bool = True,
            refine_speed_um_s: float = 100.0,
            refine_backstep_um: float = 100.0,
            backoff_um: float = 0,  # if None, computed from speed

    ) -> ReflectionFindingResult:

        z0 = self.zaber_lens.get_position()

        with self.daq.streaming():
            xs = self.daq.read_block(int(n_bg_samples))
            bg_mean = sum(xs) / len(xs)
            bg_std = (sum((x - bg_mean) ** 2 for x in xs) / len(xs)) ** 0.5
            threshold = float(bg_mean + n_sigma * bg_std)

            def run_scan(scan_speed: float, scan_dist: float) -> tuple[bool, float | None]:
                max_search_time = scan_dist / scan_speed
                self.zaber_lens.start_slewing_guarded(
                    speed_um_per_s=float(scan_speed),
                    max_distance_um=float(scan_dist),
                )
                t_start = time.monotonic()
                hits = 0

                try:
                    while True:
                        if cancel_cb and cancel_cb():
                            return False, None

                        if (time.monotonic() - t_start) >= max_search_time:
                            return False, None
                        v = self.daq.read_value(timeout_s=0.01)
                        if v > threshold:
                            hits += 1
                        else:
                            hits = 0

                        if hits >= n_hits:
                            # stop as soon as we decide it’s real
                            self.zaber_lens.stop_slewing()
                            return True, self.zaber_lens.get_position()
                finally:
                    self.zaber_lens.stop_slewing()



            # --- pass 1: fast scan to detect vicinity ---
            found, z_hit = run_scan(speed_um_s, max_search_distance_um)
            if not found:
                self.zaber_lens.move_abs(z0)
                return ReflectionFindingResult(found=False, z_um=None)

            if not refine:
                return ReflectionFindingResult(found=True, z_um=z_hit)

            # --- pass 2: backoff and rescan slowly for accurate plane ---
            # Choose a backoff big enough to cover detection + stop latency.
            # Rule of thumb: 50–200 ms worth of motion, depending on system.

            if refine_backstep_um is None:
                refine_backstep_um = max(20.0, speed_um_s * 0.1)  # 100 ms of travel, min 20 µm

            z_back = z_hit - refine_backstep_um
            self.zaber_lens.move_abs(z_back)

            def flush(seconds: float = 0.3):
                n = int(self.daq.sample_rate_hz * seconds)
                _ = self.daq.read_block(n)
            flush(0.3)  # throw away old samples

            # Slow scan forward a short distance (backoff + margin)
            refine_dist = refine_backstep_um + 50.0
            found2, z_refined = run_scan(refine_speed_um_s, refine_dist)
            if not found2:
                # fallback: return the coarse hit if refine fails
                return ReflectionFindingResult(found=True, z_um=z_hit-backoff_um)

            return ReflectionFindingResult(found=True, z_um=z_refined-backoff_um)
