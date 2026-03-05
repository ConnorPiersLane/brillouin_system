import time
import threading


from zaber_motion import Library, Units
from zaber_motion.ascii import Connection
from zaber_motion.ascii.axis import Axis


from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class ZaberPositionLog:
    t_perf: np.ndarray      # shape (n,), float64
    z_um: np.ndarray        # shape (n,), float64

    def interp(self, t_query: np.ndarray) -> np.ndarray:
        """Linear interpolation of z(t)."""
        if self.t_perf.size == 0:
            return np.empty_like(t_query, dtype=np.float64)
        # np.interp expects increasing x
        return np.interp(t_query, self.t_perf, self.z_um)


@dataclass()
class ZaberLensPosition:
    n: int
    um: float


class ZaberEyeLens:
    def __init__(self, port="COM5", axis_index=1):
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

        self.home()
        self.move_init()


    def home(self):
        self.axis.home()
        self.axis.wait_until_idle()

    def move_init(self):
        self.move_abs(12e3)

    def move_abs(self, position_um: float):
        self.axis.move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self.axis.wait_until_idle()

    def move_rel(self, delta_um: float):
        self.axis.move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self.axis.wait_until_idle()

    def get_position(self) -> float:
        return self.axis.get_position(Units.LENGTH_MICROMETRES)


    def set_zaber_position_by_class(self, zaber_position: ZaberLensPosition):
        if zaber_position.um is not None:
            self.move_abs(zaber_position.um)

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

                    time.sleep(0.01)  # 10 ms polling
            finally:
                # guard finished
                self._slew_guard_active = False

        t = threading.Thread(target=_guard_loop, daemon=True)
        self._slew_guard_thread = t
        t.start()

    def start_position_log(self, *, poll_s: float = 0.016) -> None:
        """
        Start a background thread that logs (perf_counter time, z position).
        poll_s sets the target polling period (e.g., 10 ms).
        """
        if self._log_active:
            raise RuntimeError("Position logging already running")

        poll_s = float(poll_s)
        if poll_s <= 0:
            raise ValueError("poll_s must be > 0")

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
                    alpha = 0.2  # calibrated
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
        self.connection.close()


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

    def set_zaber_position_by_class(self, zaber_position: ZaberLensPosition):
        if zaber_position.um is not None:
            self.move_abs(zaber_position.um)

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

    def start_position_log(self, *, poll_s: float = 0.01) -> None:
        """
        Start a background thread that logs (perf_counter time, z position).
        poll_s sets the target polling period (e.g., 10 ms).
        """
        if self._log_active:
            raise RuntimeError("Position logging already running")

        poll_s = float(poll_s)
        if poll_s <= 0:
            raise ValueError("poll_s must be > 0")

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

                    t = time.perf_counter()
                    z = float(self.get_position())

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


if __name__ == "__main__":
    import numpy as np

    z = ZaberEyeLens()

    print("\n--- Zaber get_position() latency test ---")

    n = 100
    times = []

    for _ in range(n):
        t0 = time.perf_counter()
        _ = z.get_position()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    import numpy as np

    times = np.array(times)

    print(f"mean latency : {times.mean()*1000:.2f} ms")
    print(f"median       : {np.median(times)*1000:.2f} ms")
    print(f"min          : {times.min()*1000:.2f} ms")
    print(f"max          : {times.max()*1000:.2f} ms")
    print(f"std          : {times.std()*1000:.2f} ms")

    print("\n--- Continuous polling test (2 seconds) ---")

    duration = 2.0
    ts = []
    zs = []

    t_end = time.perf_counter() + duration

    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        zpos = z.get_position()
        t1 = time.perf_counter()

        ts.append(0.5 * (t0 + t1))  # midpoint timestamp
        zs.append(zpos)

    ts = np.array(ts)

    rate = len(ts) / duration

    if len(ts) > 1:
        dt = np.diff(ts)
        print(f"samples collected : {len(ts)}")
        print(f"polling rate      : {rate:.1f} Hz")
        print(f"dt mean           : {dt.mean()*1000:.2f} ms")
        print(f"dt std            : {dt.std()*1000:.2f} ms")
        print(f"dt max            : {dt.max()*1000:.2f} ms")

    print("\nDone.")


    # ---- Configure test ----
    v_cmd = 1000.0          # µm/s (try 100, 200, 500, 1000)
    dist_um = 5000.0       # total travel for guarded slew
    poll_s = 0.0           # 0 means "as fast as get_position allows" (~63 Hz for you)

    print("\n--- Slewing test ---")
    print(f"Commanded speed: {v_cmd:.1f} µm/s")
    print(f"Guard distance : {dist_um:.1f} µm")
    print(f"Polling        : {'max speed' if poll_s <= 0 else f'{poll_s*1000:.1f} ms'}")

    # ---- Simple local logger (uses midpoint timestamps) ----
    t_list = []
    z_list = []
    lat_list = []


    stop_log = threading.Event()


    def log_loop():
        while not stop_log.is_set():
            t0 = time.perf_counter()
            pos = float(z.get_position())
            t1 = time.perf_counter()

            t_mid = 0.5 * (t0 + t1)

            t_list.append(t_mid)
            z_list.append(pos)
            lat_list.append(t1 - t0)

            if poll_s > 0:
                time.sleep(poll_s)

    # ---- Run motion + logging ----
    start_pos = float(z.get_position())

    th = threading.Thread(target=log_loop, daemon=True)
    th.start()

    t_start = time.perf_counter()
    z.start_slewing_guarded(speed_um_per_s=v_cmd, max_distance_um=dist_um)

    # wait until guard likely done (distance/speed + margin)
    t_expected = abs(dist_um / v_cmd)
    time.sleep(t_expected + 0.25)

    z.stop_slewing()
    t_stop = time.perf_counter()

    stop_log.set()
    th.join(timeout=2.0)

    end_pos = float(z.get_position())

    # ---- Convert to arrays ----
    t = np.asarray(t_list, dtype=np.float64)
    p = np.asarray(z_list, dtype=np.float64)
    lat = np.asarray(lat_list, dtype=np.float64)

    # Keep only samples within [t_start, t_stop] window (motion interval)
    if t.size > 0:
        m = (t >= t_start) & (t <= t_stop)
        t = t[m]
        p = p[m]
        lat = lat[m]

    print(f"\nCollected {t.size} samples during motion")
    if t.size < 3:
        print("Not enough samples to analyze. Try increasing dist_um.")
    else:
        # Ensure increasing timestamps for diffs/interp
        keep = np.concatenate(([True], np.diff(t) > 0))
        t = t[keep]
        p = p[keep]
        lat = lat[keep]

        dt = np.diff(t)
        dp = np.diff(p)

        v_est = dp / dt

        print("\n--- Timing / latency ---")
        print(f"get_position latency mean : {lat.mean()*1000:.2f} ms")
        print(f"get_position latency std  : {lat.std()*1000:.2f} ms")
        print(f"effective poll rate       : {1.0/dt.mean():.1f} Hz (mean dt {dt.mean()*1000:.2f} ms)")

        print("\n--- Velocity estimate from log ---")
        print(f"v_est mean : {v_est.mean():.2f} µm/s")
        print(f"v_est std  : {v_est.std():.2f} µm/s")
        print(f"v_cmd      : {v_cmd:.2f} µm/s")

        # Ignore first/last 15% to reduce accel/decel effects (simple trim)
        n = v_est.size
        i0 = int(0.15 * n)
        i1 = int(0.85 * n) if int(0.85 * n) > i0 else n
        v_mid = v_est[i0:i1] if i1 > i0 else v_est

        print("\n--- Mid-run velocity (trim accel/decel) ---")
        print(f"v_mid mean : {v_mid.mean():.2f} µm/s")
        print(f"v_mid std  : {v_mid.std():.2f} µm/s")

        # Compare against pos+speed prediction using first sample
        z_pred = p[0] + v_cmd * (t - t[0])
        err = p - z_pred

        print("\n--- pos+speed prediction residual ---")
        print(f"err mean : {err.mean():.2f} µm")
        print(f"err std  : {err.std():.2f} µm")
        print(f"err max  : {np.max(np.abs(err)):.2f} µm")

    print("\n--- Start/End position sanity ---")
    print(f"start_pos: {start_pos:.2f} µm")
    print(f"end_pos  : {end_pos:.2f} µm")
    print(f"delta    : {end_pos - start_pos:.2f} µm (expected ~{dist_um:+.2f} µm)")

    z.close()