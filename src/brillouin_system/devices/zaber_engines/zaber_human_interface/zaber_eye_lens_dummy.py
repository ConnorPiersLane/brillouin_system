import numpy as np
import time
import threading


from zaber_motion import Library, Units
from zaber_motion.ascii import Connection
from zaber_motion.ascii.axis import Axis


from dataclasses import dataclass
import numpy as np


from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberPositionLog


class ZaberEyeLensDummy:
    def __init__(self, port="COM5", axis_index=1):
        self.port = port
        self.axis_index = axis_index
        self._position = 100.0
        self.homed = False

        # Guard state (mirror real class)
        self._slew_guard_active = False
        self._slew_guard_thread = None
        self._slew_guard_start_pos = None
        self._slew_guard_max_dist = None  # in µm


        self._log_active = False
        self._log_thread = None
        self._log_lock = threading.Lock()
        self._log_t = []
        self._log_z = []

        print(f"[ZaberEyeLensDummy] Initialized on port {port}, axis {axis_index}")
        self.home()

    def home(self):
        self._position = 100.0
        self.homed = True
        print("[ZaberEyeLensDummy] Homed. Position reset to 100.0 µm")

    def move_abs(self, position_um: float):
        print(f"[ZaberEyeLensDummy] Moving absolute → {position_um:.2f} µm")
        self._position = position_um

    def move_rel(self, delta_um: float):
        print(f"[ZaberEyeLensDummy] Moving relative → {delta_um:+.2f} µm")
        self._position += delta_um

    def get_position(self) -> float:
        print(f"[ZaberEyeLensDummy] Current position = {self._position:.2f} µm")
        return self._position


    # ---------------- Slewing (non-blocking) ----------------
    def start_slewing(self, speed_um_per_s: float):
        direction = "forward" if speed_um_per_s >= 0 else "backward"
        print(f"[ZaberEyeLensDummy] Start slewing {direction} at {abs(speed_um_per_s):.1f} µm/s")

    def stop_slewing(self):
        print("[ZaberEyeLensDummy] Stop slewing")
        # Also stop any guard loop
        self._slew_guard_active = False

    # ---------------- Guarded slewing ----------------
    def start_slewing_guarded(self, speed_um_per_s: float, max_distance_um: float):
        """
        Simulated: Start continuous motion at speed_um_per_s (µm/s), non-blocking,
        but automatically stop once travelled distance exceeds max_distance_um.
        """
        if max_distance_um <= 0:
            print("[ZaberEyeLensDummy] Guarded slewing requested with zero/negative distance — ignoring.")
            return

        direction = "forward" if speed_um_per_s >= 0 else "backward"
        print(
            f"[ZaberEyeLensDummy] Start GUARDED slewing {direction} at "
            f"{abs(speed_um_per_s):.1f} µm/s, max distance {abs(max_distance_um):.1f} µm"
        )

        # Setup guard state
        self._slew_guard_start_pos = self._position
        self._slew_guard_max_dist = abs(max_distance_um)
        self._slew_guard_active = True

        # Start slewing (simulated, no real hardware)
        self.start_slewing(speed_um_per_s)

        def _guard_loop():
            try:
                while self._slew_guard_active:
                    # Simulate motion over 10 ms
                    dt = 0.01  # seconds
                    step = speed_um_per_s * dt
                    self._position += step

                    travelled = abs(self._position - self._slew_guard_start_pos)
                    if travelled >= self._slew_guard_max_dist:
                        print(
                            f"[ZaberEyeLensDummy] Slew GUARD triggered — "
                            f"limit {self._slew_guard_max_dist:.1f} µm reached."
                        )
                        break

                    time.sleep(dt)
            finally:
                # Ensure "motor" stops
                self.stop_slewing()

        t = threading.Thread(target=_guard_loop, daemon=True)
        self._slew_guard_thread = t
        t.start()

    def start_position_log(self, *, poll_s: float = 0.016, alpha=0.25) -> None:
        """
        Start a background thread that logs (perf_counter time, z position).
        poll_s sets the target polling period (e.g., 10 ms).
        """
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be 0<=alpha<=1 but is {alpha}")

        if self._log_active:
            raise RuntimeError("Position logging already running")

        if poll_s <= 0.015:
            raise ValueError(f"poll_s must be > 0.015 (Max value is 0.016) but is {poll_s}")

        # reset buffers
        with self._log_lock:
            self._log_t = []
            self._log_z = []

        self._log_active = True

        def _loop():
            next_t = time.perf_counter()
            try:
                while self._log_active:
                    # pace ourselves (more stable than time.sleep(poll_s) in a loop)
                    now = time.perf_counter()
                    if now < next_t:
                        time.sleep(next_t - now)
                        continue
                    next_t += poll_s

                    t0 = time.perf_counter()
                    z = self.get_position()
                    t1 = time.perf_counter()

                    lat = t1 - t0
                    alpha = 0.25  # calibrated
                    t = t0 + alpha * lat

                    with self._log_lock:
                        self._log_t.append(t)
                        self._log_z.append(z)
            finally:
                self._log_active = False

        th = threading.Thread(target=_loop, name="ZaberPosLog", daemon=True)
        self._log_thread = th
        th.start()

    def stop_position_log(self, *, join_timeout_s: float = 2.0) -> ZaberPositionLog:
        """
        Stop the position log thread and return a ZaberPositionLog.
        """
        if not self._log_active:
            # return empty
            return ZaberPositionLog(
                t_perf=np.empty(0, dtype=np.float64),
                z_um=np.empty(0, dtype=np.float64),
            )

        self._log_active = False
        th = self._log_thread
        if th is not None:
            th.join(timeout=float(join_timeout_s))
            if th.is_alive():
                raise TimeoutError("Zaber position log thread did not stop")

        with self._log_lock:
            t = np.asarray(self._log_t, dtype=np.float64)
            z = np.asarray(self._log_z, dtype=np.float64)

        # Ensure strictly increasing times for interpolation
        if t.size >= 2:
            keep = np.concatenate(([True], np.diff(t) > 0))
            t = t[keep]
            z = z[keep]

        return ZaberPositionLog(t_perf=t, z_um=z)

    from contextlib import contextmanager

    @contextmanager
    def slewing_with_log(self, speed_um_per_s: float, max_distance_um: float, *, poll_s: float = 0.01):
        """
        Context manager: starts guarded slewing + position log, then always stops both.
        Yields nothing; collect log with stop_position_log() after exiting.
        """
        self.start_position_log(poll_s=poll_s)
        try:
            self.start_slewing_guarded(speed_um_per_s, max_distance_um)
            yield
        finally:
            try:
                self.stop_slewing()
            except Exception:
                pass
            # stop log after motion stop (captures stop point)
            _ = self.stop_position_log()


    def close(self):
        print(f"[ZaberEyeLensDummy] Shutdown: port {self.port}, axis {self.axis_index}")