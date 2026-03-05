from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np

from brillouin_system.devices.ni.ni6008 import NI6008, ReadResult
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens


@dataclass(frozen=True, slots=True)
class SlewScanResult:
    daq: "ReadResult"                  # from NI
    z_t: np.ndarray                    # perf times for zaber samples
    z_um: np.ndarray                   # zaber positions
    peak_index: int                    # index into daq.values
    peak_time_perf: float              # perf time of peak sample
    peak_z_um: float                   # interpolated z at peak time
    peak_value: float                  # daq peak value


class ZaberNISlewScanner:
    """
    Runs a guarded Zaber slew while acquiring NI samples, then estimates reflection-plane
    position by mapping DAQ peak time -> Zaber position.

    Also supports a simple manual alignment offset once you know the "true" reflection plane.
    """

    def __init__(self, ni: "NI6008", zaber: "ZaberEyeLens"):
        self.ni = ni
        self.zaber = zaber

        # Manual calibration: add this to interpolated z to match your known plane
        self.z_offset_um = 0.0

    # ---------- internal helpers ----------

    @staticmethod
    def _interp_z(t_query: float, t_z: np.ndarray, z: np.ndarray) -> float:
        if t_z.size < 2:
            return float("nan")
        # np.interp needs increasing x
        return float(np.interp(np.array([t_query], dtype=np.float64), t_z, z)[0])

    @staticmethod
    def _argmax_abs(x: np.ndarray) -> int:
        return int(np.argmax(np.abs(x))) if x.size else -1

    # ---------- main scan ----------

    def run_slew_scan(
        self,
        *,
        speed_um_s: float,
        max_distance_um: float,
        max_samples: int,
        z_poll_s: float = 0.016,
        peak_mode: str = "max",   # "max" or "absmax"
    ) -> SlewScanResult:
        """
        Execute: start DAQ -> start Zaber logging -> start guarded slew -> wait -> stop -> analyze.

        Parameters:
          speed_um_s: commanded slew speed
          max_distance_um: guard distance (slew will stop after this travel)
          max_samples: NI preallocated samples (hard cap)
          z_poll_s: if 0, poll as fast as get_position allows (~63 Hz for you)
          peak_mode: "max" finds argmax(signal), "absmax" finds argmax(abs(signal))
        """
        # --- start DAQ acquisition ---
        self.ni.flush()
        self.ni.start_acquiring(max_samples=int(max_samples), chunk_size=2048)

        # --- start Zaber position logging (midpoint timestamps) ---
        # Expect you implemented these:
        #   zaber.start_position_log(poll_s=...)
        #   zaber.stop_position_log() -> ZaberPositionLog with fields t_perf, z_um
        self.zaber.start_position_log(poll_s=float(z_poll_s) if z_poll_s > 0 else 0.0)

        # --- start motion ---
        t_move_start = time.perf_counter()
        self.zaber.start_slewing_guarded(speed_um_s, max_distance_um)

        # Wait roughly for move to finish (distance/speed + margin)
        t_expected = abs(max_distance_um / speed_um_s) if speed_um_s != 0 else 0.0
        time.sleep(t_expected + 0.25)

        # --- stop motion & logging ---
        try:
            self.zaber.stop_slewing()
        except Exception:
            pass
        zlog = self.zaber.stop_position_log()

        # --- stop DAQ acquisition ---
        daq = self.ni.stop_acquiring()

        # --- analyze DAQ peak ---
        if daq.values.size == 0:
            raise RuntimeError("DAQ returned no samples; increase max_samples or check acquisition.")

        if peak_mode == "absmax":
            peak_i = self._argmax_abs(daq.values)
        else:
            peak_i = int(np.argmax(daq.values))

        peak_val = float(daq.values[peak_i])
        peak_t = daq.time_of(peak_i)

        # --- interpolate Z at DAQ peak time ---
        t_z = np.asarray(zlog.t_perf, dtype=np.float64)
        z_um = np.asarray(zlog.z_um, dtype=np.float64)

        # Ensure increasing times
        if t_z.size >= 2:
            keep = np.concatenate(([True], np.diff(t_z) > 0))
            t_z = t_z[keep]
            z_um = z_um[keep]

        peak_z = self._interp_z(peak_t, t_z, z_um) + float(self.z_offset_um)

        return SlewScanResult(
            daq=daq,
            z_t=t_z,
            z_um=z_um,
            peak_index=peak_i,
            peak_time_perf=peak_t,
            peak_z_um=peak_z,
            peak_value=peak_val,
        )

    # ---------- manual alignment / calibration ----------

    def calibrate_plane_offset(self, measured_peak_z_um: float, true_plane_z_um: float) -> float:
        """
        If you know the true reflection-plane position (e.g., from a manual procedure),
        compute and store an offset so future scans report corrected peak_z_um.

        After calling this:
          corrected_z = interpolated_z + z_offset_um
        """
        self.z_offset_um = float(true_plane_z_um) - float(measured_peak_z_um)
        return self.z_offset_um

if __name__ ==  "__main__":
    ni = NI6008(sample_rate_hz=1000)
    zaber = ZaberEyeLens()
    zaber.move_abs(10000)
    scanner = ZaberNISlewScanner(ni, zaber)

    with ni.streaming():
        res = scanner.run_slew_scan(
            speed_um_s=5000.0,
            max_distance_um=5000.0,
            max_samples=int(ni.sample_rate_hz * 15),  # enough samples for the whole motion
            z_poll_s=0.016,                             # as fast as possible (~63 Hz)
            peak_mode="max",
        )

    print("peak value:", res.peak_value)
    print("peak z (uncalibrated):", res.peak_z_um)
    z_offset = scanner.calibrate_plane_offset(measured_peak_z_um=res.peak_z_um, true_plane_z_um=13291)
    print(z_offset)
