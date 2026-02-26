from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Callable

import numpy as np

from brillouin_system.devices.ni.ni6008 import NI6008
from brillouin_system.devices.ni.ni_dummy import NIDummy
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig

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

    @staticmethod
    def parabola_peak_3pt(zs, vals) -> float:
        """
        zs, vals: arrays
        Uses max point and immediate neighbors
        """
        zs = np.asarray(zs)
        vals = np.asarray(vals)

        i = int(np.argmax(vals))

        # guard edges
        if i == 0 or i == len(vals) - 1:
            return float(zs[i])

        z1, z2, z3 = zs[i - 1], zs[i], zs[i + 1]
        v1, v2, v3 = vals[i - 1], vals[i], vals[i + 1]

        d = z2 - z1
        denom = (v1 - 2 * v2 + v3)

        if denom >= 0:
            # not concave → fallback
            return float(z2)

        dz = 0.5 * d * (v1 - v3) / denom
        dz = float(np.clip(dz, -d, d))  # safety

        return float(z2 + dz)

    def __init__(self,
                 daq: NI6008 | NIDummy,
                 zaber_axis: ZaberEyeLens,
                 *,
                 scanning_config: ScanningConfig,
                 cancel_cb: Optional[Callable[[], bool]] = None,
                 ):
        self.daq: NI6008 | NIDummy = daq
        self.zaber_lens: ZaberEyeLens = zaber_axis
        self.cancel_cb: Optional[Callable[[], bool]] = cancel_cb

        # parameters slew
        self._n_sigma: int = scanning_config.n_sigma
        self._speed_um_s: float = scanning_config.speed_um_s
        self._max_search_distance_um: float = scanning_config.max_search_distance_um
        self._n_bg_samples: int = scanning_config.n_bg_samples
        self._backstep_after_search_um: float = scanning_config.backstep_after_search_um

        # parameters refine
        self._do_refine = scanning_config.do_refine
        self._n_avg_samples: int = scanning_config.n_avg_samples
        self._step_um = scanning_config.step_um
        self._range_um = scanning_config.range_um





    def run_scan(self, scan_speed: float, scan_dist: float, threshold: float) -> tuple[bool, float | None]:
        max_search_time = scan_dist / scan_speed
        self.zaber_lens.start_slewing_guarded(
            speed_um_per_s=float(scan_speed),
            max_distance_um=float(scan_dist),
        )
        t_start = time.monotonic()
        try:
            while True:
                if self.cancel_cb and self.cancel_cb():
                    return False, None

                if (time.monotonic() - t_start) >= max_search_time:
                    return False, None

                samples = self.daq.read_available_block()

                if samples is None or len(samples) == 0:
                    continue

                arr = np.asarray(samples, dtype=float)
                cross = np.where(arr > threshold)[0]

                if cross.size > 0:
                    idx = int(cross[0])  # first threshold crossing in this chunk
                    pos_now = self.zaber_lens.get_position()
                    self.zaber_lens.stop_slewing()
                    dt = (len(arr) - idx) / self.daq.sample_rate_hz  # + 0.0 * (t_after - t_before)
                    z_hit = pos_now - dt * scan_speed
                    return True, z_hit
        finally:
            self.zaber_lens.stop_slewing()

    def measure_at(self, z_um: float) -> float:

        self.zaber_lens.move_abs(float(z_um))
        self.daq.flush()
        _ = self.daq.read_block(3)
        xs = self.daq.read_block(int(self._n_avg_samples))
        return float(np.mean(xs))

    def sample_peak_profile(
            self,
            z_hit: float,
            step_um: float,
            range_um: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample DAQ signal at evenly spaced Z positions around z_hit.

        Positions are centered so the sweep covers:
          z_hit - (n_points-1)/2 * step_um  ...  z_hit + (n_points-1)/2 * step_um

        Returns:
          zs (um), vals (V)
        """

        n_points = int(range_um / step_um) + 1
        half_span = 0.5 * (n_points - 1) * step_um

        z_start = float(z_hit - half_span)

        zs = np.empty(n_points, dtype=float)
        vals = np.empty(n_points, dtype=float)

        for i in range(n_points):
            z = z_start + i * step_um
            v = self.measure_at(z)
            zs[i] = z
            vals[i] = v

        return zs, vals

    def find_reflection_plane(
            self,
    ) -> ReflectionFindingResult:
        n_bg_samples = self._n_bg_samples
        n_sigma = self._n_sigma
        speed_um_s = self._speed_um_s
        max_search_distance_um = self._max_search_distance_um


        with self.daq.streaming():
            bg = np.asarray(self.daq.read_block(int(n_bg_samples)))
            threshold = float(bg.mean() + n_sigma * bg.std())

            t0 = time.monotonic()
            found, z_hit = self.run_scan(scan_speed=speed_um_s,
                                         scan_dist=max_search_distance_um,
                                         threshold=threshold)

            t1 = time.monotonic()

            if not found or z_hit is None:
                return ReflectionFindingResult(found=False, z_um=None)

            z_hit = float(z_hit - self._backstep_after_search_um)

            if not self._do_refine:
                return ReflectionFindingResult(found=True, z_um=z_hit)

            zs, vals = self.sample_peak_profile(z_hit, step_um=self._step_um, range_um=self._range_um)

            z_peak = self.parabola_peak_3pt(zs, vals)
            t2 = time.monotonic()


            # Just for testing:

            print(f"time scanning: {t1-t0}")
            print(f"time stepping: {t2 - t1}")

            # Print nicely
            print("i\tz_um\tvalue_V")
            for i, (z, v) in enumerate(zip(zs, vals)):
                print(f"{i}\t{z:.2f}\t{v:.6f}")

            # Optional: also print quick summary
            imax = int(np.argmax(vals))
            print(f"\nMax: i={imax}, z={zs[imax]:.2f} um, v={vals[imax]:.6f} V")
            zs = [float(xx - zs[int(len(zs)/2)]) for xx in zs]
            vals = [float(v) for v in vals]
            print('Coarse structure')
            print(zs)
            print(vals)

            print('Fine structure')
            zs, vals = self.sample_peak_profile(z_hit, step_um=2, range_um=100)
            zs = [float(xx - zs[int(len(zs)/2)]) for xx in zs]
            vals = [float(v) for v in vals]
            print(zs)
            print(vals)


            return ReflectionFindingResult(found=True, z_um=z_peak)

