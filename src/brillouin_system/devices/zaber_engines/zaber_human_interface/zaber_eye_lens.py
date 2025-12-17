import time
import threading
from dataclasses import dataclass

from zaber_motion import Library, Units
from zaber_motion.ascii import Connection
from zaber_motion.ascii.axis import Axis



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

    def close(self):
        print(f"[ZaberEyeLensDummy] Shutdown: port {self.port}, axis {self.axis_index}")


if __name__ == "__main__":
    zaber_hi = ZaberEyeLens()