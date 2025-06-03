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
            position_um:

        Returns:

        """

        self.axis_map[which_axis].move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self.axis_map[which_axis].wait_until_idle()

    def move_rel(self, which_axis: str, delta_um):
        """
        Args:
            which_axis: 'x', 'y', 'z'
        """
        self.axis_map[which_axis].move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self.axis_map[which_axis].wait_until_idle()

    def get_position(self, which_axis: str):
        """
        Args:
            which_axis: 'x', 'y', 'z'
        """
        return self.axis_map[which_axis].get_position(Units.LENGTH_MICROMETRES)

    def set_speed(self, which_axis: str, speed_mm_per_s):
        """
        Args:
            which_axis: 'x', 'y', 'z'
        """
        self.axis_map[which_axis].settings.set('maxspeed', speed_mm_per_s, Units.VELOCITY_MILLIMETRES_PER_SECOND)

    def set_acceleration(self, which_axis, accel_native_units):
        """
        Args:
            which_axis: 'x', 'y', 'z'
        """
        self.axis_map[which_axis].settings.set('accel', accel_native_units)

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

