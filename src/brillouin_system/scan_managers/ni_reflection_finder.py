from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Callable

import numpy as np

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
            bg = np.asarray(self.daq.read_block(int(n_bg_samples)))
            threshold = float(bg.mean() + n_sigma * bg.std())
            dt_sample = 1 / self.daq.sample_rate_hz


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

                        samples = self.daq.read_available_block()
                        if not samples:
                            continue
                        arr = np.asarray(samples, dtype=float)
                        cross = np.where(arr > threshold)[0]

                        if cross.size > 0:
                            idx = int(cross[0])  # first threshold crossing in this chunk
                            hits += 1
                        else:
                            hits = 0

                        if hits >= n_hits:
                            # stop as soon as we decide it’s real
                            t_before = time.monotonic()
                            pos_now = self.zaber_lens.get_position()
                            t_after = time.monotonic()
                            self.zaber_lens.stop_slewing()
                            dt = (len(arr) - idx) / self.daq.sample_rate_hz + 0.5 * (t_after - t_before)
                            z_hit = pos_now - dt * scan_speed
                            return True, z_hit
                finally:
                    self.zaber_lens.stop_slewing()



            # --- pass 1: fast scan to detect vicinity ---
            found, z_hit = run_scan(speed_um_s, max_search_distance_um)

            def measure_at(self, z_um: float, n_avg: int = 50, settle_s: float = 0.005) -> float:
                self.zaber_lens.move_abs(float(z_um))
                time.sleep(settle_s)
                self.daq.flush()
                xs = self.daq.read_block(int(n_avg))
                return float(np.mean(xs))

            def sample_peak_profile(
                    z_hit: float,
                    *,
                    n_points: int = 100,
                    step_um: float = 5.0,
                    n_avg: int = 50,
                    settle_s: float = 0.005,
            ) -> tuple[np.ndarray, np.ndarray]:
                """
                Sample DAQ signal at evenly spaced Z positions around z_hit.

                Positions are centered so the sweep covers:
                  z_hit - (n_points-1)/2 * step_um  ...  z_hit + (n_points-1)/2 * step_um

                Returns:
                  zs (um), vals (V)
                """
                half_span = 0.5 * (n_points - 1) * step_um
                z_start = float(z_hit - half_span)

                zs = np.empty(n_points, dtype=float)
                vals = np.empty(n_points, dtype=float)

                for i in range(n_points):
                    z = z_start + i * step_um
                    v = measure_at(z, n_avg=n_avg, settle_s=settle_s)
                    zs[i] = z
                    vals[i] = v

                # Print nicely
                print("i\tz_um\tvalue_V")
                for i, (z, v) in enumerate(zip(zs, vals)):
                    print(f"{i}\t{z:.2f}\t{v:.6f}")

                # Optional: also print quick summary
                imax = int(np.argmax(vals))
                print(f"\nMax: i={imax}, z={zs[imax]:.2f} um, v={vals[imax]:.6f} V")

                return zs, vals

            zs, vals = sample_peak_profile(z_hit)
            print(zs)
            print(vals)

            if not found:
                return ReflectionFindingResult(found=False, z_um=None)

            if not refine:
                return ReflectionFindingResult(found=True, z_um=z_hit)

            # --- pass 2: backoff and rescan slowly for accurate plane ---
            # Choose a backoff big enough to cover detection + stop latency.
            # Rule of thumb: 50–200 ms worth of motion, depending on system.



            return ReflectionFindingResult(found=True, z_um=z_hit)
