from zaber_motion import Library, Units
from zaber_motion.ascii import Connection
from brillouin_system.my_dataclasses.zaber_position import ZaberPosition


class ZaberAxisController:
    def __init__(self, x_axis=None, y_axis=None, z_axis=None):
        self.axes = {'x': x_axis, 'y': y_axis, 'z': z_axis}

    def move_abs(self, axis, pos_um):
        if self.axes[axis] is not None:
            self.axes[axis].move_absolute(pos_um, Units.LENGTH_MICROMETRES).wait_until_idle()

    def move_rel(self, axis, delta_um):
        if self.axes[axis] is not None:
            self.axes[axis].move_relative(delta_um, Units.LENGTH_MICROMETRES).wait_until_idle()

    def get_position(self, axis):
        if self.axes[axis] is not None:
            return self.axes[axis].get_position(Units.LENGTH_MICROMETRES)

    def set_speed(self, axis, speed_mm_s):
        if self.axes[axis] is not None:
            self.axes[axis].settings.set('maxspeed', speed_mm_s, Units.VELOCITY_MILLIMETRES_PER_SECOND)

    def set_acceleration(self, axis, accel_native):
        if self.axes[axis] is not None:
            self.axes[axis].settings.set('accel', accel_native)

    def get_position_class(self):
        return ZaberPosition(
            x=self.get_position('x'),
            y=self.get_position('y'),
            z=self.get_position('z')
        )

    def set_position_class(self, position: ZaberPosition):
        if position.x is not None:
            self.move_abs('x', position.x)
        if position.y is not None:
            self.move_abs('y', position.y)
        if position.z is not None:
            self.move_abs('z', position.z)

    def home_all(self):
        for axis in self.axes.values():
            if axis:
                axis.home().wait()


class ZaberLightController:
    def __init__(self, light_axes: dict):
        """
        light_axes: dict like {'white': Axis, 'blue': Axis, ...}
        """
        self.lights = light_axes
        self.max_currents = self._read_max_currents()

    def _read_max_currents(self):
        max_vals = {}
        for color, lamp in self.lights.items():
            reply = lamp.send("get lamp.current.max")
            max_vals[color] = float(reply.data)
        return max_vals

    def set_light(self, color, state: bool):
        command = "lamp on" if state else "lamp off"
        self.lights[color].send(command)

    def set_power(self, color, percent):
        current = percent * self.max_currents[color] / 100
        self.lights[color].send(f"set lamp.current {current}")

    def get_power(self, color):
        reply = self.lights[color].send("get lamp.current")
        return 100 * float(reply.data) / self.max_currents[color]

    def set_initial_lighting(self):
        self.set_light('trans', True)
        for c in ['white', 'blue', 'red']:
            self.set_light(c, False)


class ZaberFilterController:
    def __init__(self, filter_device):
        self.device = filter_device
        self.device.home().wait()

    def move_to_index(self, index: int):
        self.device.send(f"move index {index}")


class ZaberMicroscopeController:
    def __init__(self, port="COM4"):
        Library.enable_device_db_store()
        self.connection = Connection.open_serial_port(port)
        devices = self.connection.detect_devices()

        if len(devices) < 5:
            raise RuntimeError("Expected at least 5 Zaber devices for full microscope.")

        self.light_device = devices[1]
        self.xy_device = devices[2]
        self.z_device = devices[3]
        self.filter_device = devices[4]

        # Axes
        x_axis = self.xy_device.get_axis(2)
        y_axis = self.xy_device.get_axis(1)
        z_axis = self.z_device.get_axis(1)

        # Lights
        light_axes = {
            'white': self.light_device.get_axis(1),
            'blue': self.light_device.get_axis(2),
            'red': self.light_device.get_axis(3),
            'trans': self.light_device.get_axis(4)
        }

        # Sub-controllers
        self.motion = ZaberAxisController(x_axis, y_axis, z_axis)
        self.lighting = ZaberLightController(light_axes)
        self.filters = ZaberFilterController(self.filter_device)

        self.motion.home_all()
        self.lighting.set_initial_lighting()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
