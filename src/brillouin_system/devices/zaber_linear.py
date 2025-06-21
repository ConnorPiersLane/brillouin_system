from zaber_motion import Library, Units
from zaber_motion.ascii import Connection

from brillouin_system.my_dataclasses.zaber_position import ZaberPosition




class ZaberLinearController:
    def __init__(self, port="COM5", axis_index=1):
        Library.enable_device_db_store()
        self.connection = Connection.open_serial_port(port)
        devices = self.connection.detect_devices()
        if not devices:
            raise RuntimeError("No Zaber devices found.")
        self.x_axis = devices[0].get_axis(axis_index)
        self.y_axis = None
        self.z_axis = None

        self.axis_map = {
            'x': self.x_axis,
            'y': self.y_axis,
            'z': self.z_axis,
        }

        self.home()

    def home(self):
        self.x_axis.home()
        self.x_axis.wait_until_idle()

    def move_abs(self, which_axis: str, position_um: float):
        """

        Args:
            which_axis: 'x', 'y', 'z'
            position_um: [um]

        Returns:

        """

        self.axis_map[which_axis].move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self.axis_map[which_axis].wait_until_idle()

    def move_rel(self, which_axis: str, delta_um: float):
        """
        Args:
            which_axis: 'x', 'y', 'z'
            delta_um: [um]
        """
        self.axis_map[which_axis].move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self.axis_map[which_axis].wait_until_idle()

    def get_position(self, which_axis: str):
        """
        Args:
            which_axis: 'x', 'y', 'z'
        """
        return self.axis_map[which_axis].get_position(Units.LENGTH_MICROMETRES)


    def get_zaber_position_class(self) -> ZaberPosition:
        return ZaberPosition(x=self.get_position('x'), y=0, z=0)

    def set_zaber_position_by_class(self, zaber_position: ZaberPosition):
        axes = self.get_available_axes()
        if zaber_position.x is not None and 'x' in axes:
            self.move_abs('x', zaber_position.x)
        if zaber_position.y is not None and 'y' in axes:
            self.move_abs('y', zaber_position.y)
        if zaber_position.z is not None and 'z' in axes:
            self.move_abs('z', zaber_position.z)

    def get_available_axes(self) -> list[str]:
        """Return list of available axes as string labels: ['x', 'y']"""
        return [label for label, axis in self.axis_map.items() if axis is not None]

    def close(self):
        self.connection.close()

class ZaberLinearDummy:
    def __init__(self, port="COM5", axis_index=1):
        self.port = port
        self.axis_index = axis_index
        self._positions = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0
        }
        self.speed_mm_per_s = 10.0
        self.accel_native_units = 600
        self.homed = False
        print(f"[ZaberDummy] Initialized on port {port}, axis {axis_index}")

        self.home()

    def _check_axis(self, which_axis: str):
        if which_axis not in self._positions:
            raise ValueError(f"[ZaberDummy] Axis '{which_axis}' is not valid.")

    def home(self):
        for axis in self._positions:
            self._positions[axis] = 0.0
        self.homed = True
        print("[ZaberDummy] Homed. All positions reset to 0.0 µm")

    def move_abs(self, which_axis: str, position_um: float):
        self._check_axis(which_axis)
        print(f"[ZaberDummy] Moving {which_axis.upper()} absolute → {position_um:.2f} µm")
        self._positions[which_axis] = position_um

    def move_rel(self, which_axis: str, delta_um: float):
        self._check_axis(which_axis)
        print(f"[ZaberDummy] Moving {which_axis.upper()} relative → {delta_um:+.2f} µm")
        self._positions[which_axis] += delta_um

    def get_position(self, which_axis: str):
        self._check_axis(which_axis)
        pos = self._positions[which_axis]
        print(f"[ZaberDummy] Current {which_axis.upper()} position = {pos:.2f} µm")
        return pos


    def get_zaber_position_class(self) -> ZaberPosition:
        return ZaberPosition(
            x=self._positions['x'],
            y=self._positions['y'],
            z=self._positions['z']
        )

    def get_available_axes(self) -> list[str]:
        return list(self._positions.keys())

    def set_zaber_position_by_class(self, zaber_position: ZaberPosition):
        axes = self.get_available_axes()
        if zaber_position.x is not None and 'x' in axes:
            self.move_abs('x', zaber_position.x)
        if zaber_position.y is not None and 'y' in axes:
            self.move_abs('y', zaber_position.y)
        if zaber_position.z is not None and 'z' in axes:
            self.move_abs('z', zaber_position.z)

    def close(self):
        print(f"[ZaberDummy] Shutdown: port {self.port}, axis {self.axis_index}")
