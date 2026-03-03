from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Callable

import numpy as np


from brillouin_system.devices.ni.ni_dummy import NIBase
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
                 daq: NIBase,
                 zaber_axis: ZaberEyeLens,
                 *,
                 scanning_config: ScanningConfig,
                 cancel_cb: Optional[Callable[[], bool]] = None,
                 ):
        self.daq: NIBase = daq

        self.zaber_lens: ZaberEyeLens = zaber_axis
        self.cancel_cb: Optional[Callable[[], bool]] = cancel_cb



        # parameters slew
        self._n_sigma: int = scanning_config.n_sigma
        self._speed_um_s: float = scanning_config.speed_um_s
        if self._speed_um_s == 0:
            raise ValueError("scan_speed must be nonzero")
        self._max_search_distance_um: float = scanning_config.max_search_distance_um
        self._n_bg_samples: int = self.acquisition_time_to_samples(scanning_config.background_acquisition_time_ms)
        self._backstep_after_search_um: float = scanning_config.backstep_after_search_um

        # parameters refine
        self._do_refine = scanning_config.do_refine
        self._n_point_samples: int = self.acquisition_time_to_samples(scanning_config.point_acquisition_time_ms)
        self._step_um = scanning_config.step_um
        self._range_um = scanning_config.range_um
        self._n_max_values: int = scanning_config.n_max_values

    def acquisition_time_to_samples(self, acq_time_ms: float) -> int:
        n = int(round(acq_time_ms * 1e-3 * self.daq.get_sample_rate()))
        return max(1, n)

    def max_values_mean(self, xs) -> float:
        xs = np.asarray(xs)  # cheap if already ndarray
        s = xs.size
        n = self._n_max_values

        if s == 0:
            raise ValueError('Input Array has size 0')
        if s <= n:
            return float(xs.mean())
        return float(np.partition(xs, s - n)[s - n:].mean())

    def run_scan(self, scan_speed: float, scan_dist: float, threshold: float) -> tuple[bool, float | None]:
        max_search_time = abs(scan_dist / scan_speed)
        self.zaber_lens.start_slewing_guarded(
            speed_um_per_s=float(scan_speed),
            max_distance_um=float(abs(scan_dist)),
        )

        t_start = time.monotonic()
        try:
            while True:
                if self.cancel_cb and self.cancel_cb():
                    log.info(f"[Reflection Finding] Cancelled during scanning.")
                    return False, None

                if (time.monotonic() - t_start) >= max_search_time:
                    return False, None

                samples = self.daq.read_available_block()
                if samples.size == 0:
                    continue

                arr = samples  # already np.ndarray
                cross = np.where(arr > threshold)[0]

                if cross.size > 0:
                    i0 = cross[0]
                    i_peak = i0 + np.argmax(arr[i0:])
                    pos_now = self.zaber_lens.get_position()
                    self.zaber_lens.stop_slewing()
                    dt = (len(arr) - i_peak) / self.daq.get_sample_rate()  # + 0.0 * (t_after - t_before)
                    z_hit = pos_now - dt * scan_speed
                    return True, float(z_hit)
        finally:
            self.zaber_lens.stop_slewing()

    def measure_at(self, z_um: float) -> float:
        self.zaber_lens.move_abs(float(z_um))
        self.daq.flush()
        xs = self.daq.read_block(self._n_point_samples)
        return self.max_values_mean(xs)

    def sample_peak_profile(
            self,
            z_hit: float,
            step_um: float,
            range_um: float,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
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

            if self.cancel_cb and self.cancel_cb():
                log.info(f"[Reflection Finding] Cancelled during scanning.")
                return None, None

            z = z_start + i * step_um
            v = self.measure_at(z)
            zs[i] = z
            vals[i] = v

        return zs, vals

    def find_reflection_plane(
            self,
            is_go_forwards: bool = True
    ) -> ReflectionFindingResult:
        n_bg_samples = self._n_bg_samples
        n_sigma = self._n_sigma
        if is_go_forwards:
            speed_um_s = self._speed_um_s
        else:
            speed_um_s = -1*self._speed_um_s
        max_search_distance_um = self._max_search_distance_um


        with self.daq.streaming():
            self.daq.flush()
            _ = self.daq.read_block(3)
            bg: np.ndarray = self.daq.read_block(int(n_bg_samples))
            threshold = float(bg.mean() + n_sigma * bg.std())

            t0 = time.monotonic()
            found, z_hit = self.run_scan(scan_speed=speed_um_s,
                                         scan_dist=max_search_distance_um,
                                         threshold=threshold)

            t1 = time.monotonic()

            if not found or z_hit is None:
                return ReflectionFindingResult(found=False, z_um=None)

            direction = 1.0 if is_go_forwards else -1.0
            z_hit = float(z_hit - direction * self._backstep_after_search_um)

            if not self._do_refine:
                return ReflectionFindingResult(found=True, z_um=z_hit)

            zs, vals = self.sample_peak_profile(z_hit, step_um=self._step_um, range_um=self._range_um)

            # Cancel Callback has been pressed by frontend
            if zs is None or vals is None:
                return ReflectionFindingResult(found=False, z_um=None)


            imax = int(np.argmax(vals))
            if vals[imax] < threshold:
                return ReflectionFindingResult(found=False, z_um=None)

            z_peak = float(zs[imax])
            z_peak2 = self.parabola_peak_3pt(zs, vals)
            t2 = time.monotonic()


            # Just for testing:
            print(f"----------------------------------------------------------------------------------------------")
            print(f"time scanning: {t1-t0}")
            print(f"time stepping: {t2 - t1}")

            # Print nicely
            print("i\tz_um\tvalue_V")
            for i, (z, v) in enumerate(zip(zs, vals)):
                print(f"{i}\t{z:.2f}\t{v:.6f}")

            # Optional: also print quick summary
            imax = int(np.argmax(vals))
            print(f"\nMax: i={imax}, z={zs[imax]:.2f} um, v={vals[imax]:.6f} V")
            zs = [float(xx) for xx in zs]
            vals = [float(v) for v in vals]
            print('Coarse structure')
            print(zs)
            print(vals)
            print(f"Peak Estimate from slewing: {z_hit}")
            print(f"Peak from max refine: {z_peak}")
            print(f"Peak from parable fit: {z_peak2}")

            print('Fine structure with dz = 1um')
            zs, vals = self.sample_peak_profile(z_hit, step_um=1, range_um=100)
            imax = int(np.argmax(vals))
            print(f"\nMax: i={imax}, z={zs[imax]:.2f} um, v={vals[imax]:.6f} V")
            zs = [float(xx) for xx in zs]
            vals = [float(v) for v in vals]
            print(zs)
            print(vals)
            print(f"----------------------------------------------------------------------------------------------")


            return ReflectionFindingResult(found=True, z_um=z_peak)

