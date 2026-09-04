import time
import threading


from zaber_motion import Library, Units
from zaber_motion.ascii import Connection
from zaber_motion.ascii.axis import Axis

import numpy as np

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_position_log import ZaberPositionLog


class ZaberEyeLens:
    def __init__(self, port="COM5", axis_index=1, home_on_connect=True):
        """
        home_on_connect=True (default) homes the axis and moves to the init
        position — the normal HI startup. Pass False to attach to the lens
        WITHOUT moving it (e.g. during calibration).
        """
        Library.enable_device_db_store()
        self.connection = Connection.open_serial_port(port)
        devices = self.connection.detect_devices()
        if not devices:
            raise RuntimeError("No Zaber devices found.")
        self.axis: Axis = devices[0].get_axis(axis_index)

        # Guard state
        self._slew_guard_active = False
        self._slew_guard_thread = None
        self._slew_guard_start_pos = None
        self._slew_guard_max_dist = None  # in µm

        self._log_active = False
        self._log_thread = None
        self._log_lock = threading.Lock()
        self._log_t = []
        self._log_z = []

        if home_on_connect:
            self.home()
            self.move_init()


    def home(self):
        self.axis.home()
        self.axis.wait_until_idle()

    def move_init(self):
        self.move_abs(10e3)
        # self.move_abs(0)

    def move_abs(self, position_um: float):
        self.axis.move_absolute(float(position_um), Units.LENGTH_MICROMETRES)
        self.axis.wait_until_idle()

    def move_rel(self, delta_um: float):
        self.axis.move_relative(float(delta_um), Units.LENGTH_MICROMETRES)
        self.axis.wait_until_idle()

    def get_position(self) -> float:
        return self.axis.get_position(Units.LENGTH_MICROMETRES)


    def start_slewing(self, speed_um_per_s: float):
        """
        Start moving continuously at the given speed (µm/s).
        Positive = forward, negative = backward.
        Non-blocking.
        """
        self.axis.move_velocity(
            speed_um_per_s,
            Units.VELOCITY_MICROMETRES_PER_SECOND
        )

    def stop_slewing(self):
        """
        Stop any ongoing velocity move and turn off the distance guard.
        Safe to call even if not currently slewing.
        """
        # Tell guard thread (if any) to stop watching
        self._slew_guard_active = False

        # Stop the axis
        self.axis.stop()
        self.axis.wait_until_idle()

    def start_slewing_guarded(self, speed_um_per_s: float, max_distance_um: float):
        """
        Start continuous motion at speed_um_per_s (µm/s), non-blocking,
        but automatically stop if the travelled distance exceeds max_distance_um.

        This returns immediately: you can do other work (e.g. grab a frame).
        Even if that work crashes, the guard thread will stop the motion
        once max_distance_um is exceeded.
        """
        if max_distance_um <= 0:
            # Nothing to guard -> just don't move
            return

        # Record starting position & allowed distance
        self._slew_guard_start_pos = self.get_position()
        self._slew_guard_max_dist = float(abs(max_distance_um))
        self._slew_guard_active = True

        # Start continuous motion (non-blocking)
        self.start_slewing(speed_um_per_s)

        # Start watchdog thread
        def _guard_loop():
            try:
                while self._slew_guard_active:
                    pos = self.get_position()
                    travelled = abs(pos - self._slew_guard_start_pos)

                    if travelled >= self._slew_guard_max_dist:
                        # exceeded allowed distance -> hard stop
                        self.axis.stop()
                        self.axis.wait_until_idle()
                        break

                    time.sleep(0.03)  # 30 ms polling
            finally:
                # guard finished
                self._slew_guard_active = False

        t = threading.Thread(target=_guard_loop, daemon=True)
        self._slew_guard_thread = t
        t.start()

    def start_position_log(self, *, poll_s: float = 0.016, alpha = 0.25) -> None:
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

    def close(self):
        self.connection.close()



if __name__ == "__main__":
    z = ZaberEyeLens()
    print(f"position: {z.get_position():.2f} um")
    z.close()
