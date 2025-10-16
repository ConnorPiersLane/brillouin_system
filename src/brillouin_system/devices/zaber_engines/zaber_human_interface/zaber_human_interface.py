from zaber_motion import Library, Units
from zaber_motion.ascii import Connection

from brillouin_system.logging_utils.logging_setup import get_logger
log = get_logger(__name__)


class ZaberHumanInterface:
    def __init__(self, port="COM6", axis_index=1):
        Library.enable_device_db_store()
        self.connection = Connection.open_serial_port(port)
        devices = self.connection.detect_devices()
        if not devices:
            raise RuntimeError("No Zaber devices found.")
        self.x_axis = devices[1].get_axis(axis_index)
        self.y_axis = devices[2].get_axis(axis_index)
        self.z_axis = devices[0].get_axis(axis_index)

        self.axis_map = {
            'x': self.x_axis,
            'y': self.y_axis,
            'z': self.z_axis,
        }

        self.home()
        self.move_to_init_position()

    def home(self):
        self.x_axis.home()
        self.y_axis.home()
        self.z_axis.home()
        self.x_axis.wait_until_idle()
        self.y_axis.wait_until_idle()
        self.z_axis.wait_until_idle()

    def move_to_init_position(self):
        self.move_abs(12e3, 10e3, 12e3)

    def move_rel(self, dx: float = None, dy: float = None, dz: float = None):
        # queue moves
        if dx is not None:
            self.axis_map['x'].move_relative(dx, Units.LENGTH_MICROMETRES, wait_until_idle=False)
        if dy is not None:
            self.axis_map['y'].move_relative(dy, Units.LENGTH_MICROMETRES, wait_until_idle=False)
        if dz is not None:
            self.axis_map['z'].move_relative(dz, Units.LENGTH_MICROMETRES, wait_until_idle=False)

        # now wait for all available axes
        for axis in ('x', 'x', 'z'):
            self.axis_map[axis].wait_until_idle()

    def move_abs(self, x: float = None, y: float = None, z: float = None):
        # queue moves
        if x is not None:
            self.axis_map['x'].move_absolute(x, Units.LENGTH_MICROMETRES, wait_until_idle=False)
        if y is not None:
            self.axis_map['y'].move_absolute(y, Units.LENGTH_MICROMETRES, wait_until_idle=False)
        if z is not None:
            self.axis_map['z'].move_absolute(z, Units.LENGTH_MICROMETRES, wait_until_idle=False)

        # now wait for all available axes
        for axis in ('x', 'x', 'z'):
            self.axis_map[axis].wait_until_idle()

    def get_position(self) -> tuple[float, float, float]:
        """
        Args:
            which_axis: 'x', 'y', 'z'
        """
        x = self.axis_map['x'].get_position(Units.LENGTH_MICROMETRES)
        y = self.axis_map['y'].get_position(Units.LENGTH_MICROMETRES)
        z = self.axis_map['z'].get_position(Units.LENGTH_MICROMETRES)
        return x,y,z


    def close(self):
        self.connection.close()

class ZaberHumanInterfaceDummy:
    def __init__(self, port="COM5", axis_index=1):
        self.port = port
        self.axis_index = axis_index
        self._positions = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0
        }
        self.homed = False
        log.info(f"[ZaberDummy] Initialized on port {port}, axis {axis_index}")

        self.home()
        self.move_to_init_position()

    def home(self):
        for axis in self._positions:
            self._positions[axis] = 0.0
        self.homed = True
        log.info("[ZaberDummy] Homed. All positions reset to 0.0 µm")

    def move_to_init_position(self):
        self.move_abs(12e3, 10e3, 12e3)

    def move_rel(self, dx: float = None, dy: float = None, dz: float = None):
        if dx is not None:
            self._positions['x'] += dx
            log.info(f"[ZaberDummy] Moving X relative → {dx:+.2f} µm (now {self._positions['x']:.2f})")
        if dy is not None:
            self._positions['y'] += dy
            log.info(f"[ZaberDummy] Moving Y relative → {dy:+.2f} µm (now {self._positions['y']:.2f})")
        if dz is not None:
            self._positions['z'] += dz
            log.info(f"[ZaberDummy] Moving Z relative → {dz:+.2f} µm (now {self._positions['z']:.2f})")

        log.info("[ZaberDummy] Relative move(s) completed.")

    def move_abs(self, x: float = None, y: float = None, z: float = None):
        if x is not None:
            self._positions['x'] = x
            log.info(f"[ZaberDummy] Moving X absolute → {x:.2f} µm")
        if y is not None:
            self._positions['y'] = y
            log.info(f"[ZaberDummy] Moving Y absolute → {y:.2f} µm")
        if z is not None:
            self._positions['z'] = z
            log.info(f"[ZaberDummy] Moving Z absolute → {z:.2f} µm")

        log.info("[ZaberDummy] Absolute move(s) completed.")

    def get_position(self) -> tuple[float, float, float]:
        x = self._positions['x']
        y = self._positions['y']
        z = self._positions['z']
        log.info(f"[ZaberDummy] Current positions → X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        return x, y, z

    def close(self):
        log.info(f"[ZaberDummy] Shutdown: port {self.port}, axis {self.axis_index}")


if __name__ == "__main__":
    zaber_hi = ZaberHumanInterface()
    zaber_hi_dummy = ZaberHumanInterfaceDummy()