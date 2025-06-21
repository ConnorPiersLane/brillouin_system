from zaber_motion import Library, Units
from zaber_motion.ascii import Connection, Axis
from zaber_motion.microscopy import Illuminator, FilterChanger, IlluminatorChannel

from brillouin_system.my_dataclasses.zaber_position import ZaberPosition


class ZaberMicroscope:
    def __init__(self, port="COM4"):
        Library.enable_device_db_store()
        self._connection = Connection.open_serial_port(port)
        self._devices = self._connection.detect_devices()

        if not self._devices:
            raise RuntimeError("No Zaber devices detected on port.")

        self._initialize_devices()
        self._initialize_axes()
        self._initialize_lights()


        self._axis_map = {
            'x': self.x_axis,
            'y': self.y_axis,
            'z': self.z_axis,
        }


        self.move_axis_home()


    def _get_device(self, address):
        for device in self._devices:
            if device.device_address == address:
                return device
        raise ValueError(f"Device with address {address} not found.")

    def _initialize_devices(self):
        self._illuminator_device = self._get_device(2)
        self._xy_stage_device = self._get_device(3)
        self._z_stage_device = self._get_device(4)
        self._filter_device = self._get_device(5)
        self.filter_changer: FilterChanger = FilterChanger(self._filter_device)

    def _initialize_axes(self):
        self.x_axis: Axis = self._xy_stage_device.get_axis(2)
        self.y_axis: Axis = self._xy_stage_device.get_axis(1)
        self.z_axis: Axis = self._z_stage_device.get_axis(1)


    def _initialize_lights(self):
        self._illuminator = Illuminator(self._illuminator_device)
        self.led_white_below: IlluminatorChannel = self._illuminator.get_channel(1)
        self.led_blue_385_below: IlluminatorChannel = self._illuminator.get_channel(2)
        self.led_red_625_below: IlluminatorChannel = self._illuminator.get_channel(3)
        self.led_white_top: IlluminatorChannel = self._illuminator.get_channel(4)

        # set lights
        self.led_white_below.off()
        self.led_blue_385_below.off()
        self.led_red_625_below.off()
        self.led_white_top.on()

    def move_filter(self, index=1):
        """
        Index 1 = brightfield (BS)
        Index 2 = fluorescence (dichroic)
        Args:
            index: int 1 or 2

        Returns: None

        """
        self.filter_changer.change(index)

    # moves Zaber stages to home position
    def move_axis_home(self, which_axis='a'):

        home_x = 60000.0
        home_y = 50000.0
        home_z = 6990.0

        if which_axis == 'x':
            self.move_abs(which_axis, home_x)
        elif which_axis == 'y':
            self.move_abs(which_axis, home_y)
        elif which_axis == 'z':
            self.move_abs(which_axis, home_z)
        elif which_axis == 'a':
            self.move_abs('x', home_x)
            self.move_abs('y', home_y)
            self.move_abs('z', home_z)


    def move_abs(self, which_axis: str, position_um: float):
        """

        Args:
            which_axis: 'x', 'y', 'z'
            position_um: [um]

        Returns:

        """

        self._axis_map[which_axis].move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self._axis_map[which_axis].wait_until_idle()

    def move_rel(self, which_axis: str, delta_um: float):
        """
        Args:
            which_axis: 'x', 'y', 'z'
            delta_um: [um]
        """
        self._axis_map[which_axis].move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self._axis_map[which_axis].wait_until_idle()

    def get_position(self, which_axis: str):
        """
        Args:
            which_axis: 'x', 'y', 'z'
        """
        return self._axis_map[which_axis].get_position(Units.LENGTH_MICROMETRES)


    def get_zaber_position_class(self) -> ZaberPosition:
        return ZaberPosition(x=self.get_position('x'), y=self.get_position('y'), z=self.get_position('z'))

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
        return [label for label, axis in self._axis_map.items() if axis is not None]

    def close(self):
        self._connection.close()


    def shutdown(self):
        print("[Microscope] Shutting down...")
        self.led_white_below.off()
        self.led_blue_385_below.off()
        self.led_red_625_below.off()
        self.led_white_top.off()
        self._connection.close()
        print("[Microscope] Shutdown complete.")

from brillouin_system.my_dataclasses.zaber_position import ZaberPosition


class DummyZaberMicroscope:
    def __init__(self, port="COM4"):
        print(f"[Dummy] Initialized mock microscope on port {port}")
        self._position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self._active_filter = 1
        self._led_states = {
            'white_below': False,
            'blue_385_below': False,
            'red_625_below': False,
            'white_top': False
        }

    def move_filter(self, index=1):
        print(f"[Dummy] Moved filter to index {index}")
        self._active_filter = index

    def move_axis_home(self, which_axis='a'):
        print(f"[Dummy] Moving axes to simulated home position ({which_axis})")
        home_pos = {'x': 60000.0, 'y': 50000.0, 'z': 6990.0}
        if which_axis == 'a':
            self._position.update(home_pos)
        elif which_axis in home_pos:
            self._position[which_axis] = home_pos[which_axis]

    def move_abs(self, which_axis: str, position_um: float):
        print(f"[Dummy] Moving {which_axis}-axis to {position_um} µm")
        self._position[which_axis] = position_um

    def move_rel(self, which_axis: str, delta_um: float):
        print(f"[Dummy] Moving {which_axis}-axis by {delta_um} µm")
        self._position[which_axis] += delta_um

    def get_position(self, which_axis: str):
        pos = self._position[which_axis]
        print(f"[Dummy] Getting {which_axis}-axis position: {pos} µm")
        return pos

    def get_zaber_position_class(self) -> ZaberPosition:
        print("[Dummy] Getting full position")
        return ZaberPosition(**self._position)

    def set_zaber_position_by_class(self, zaber_position: ZaberPosition):
        print(f"[Dummy] Setting position from ZaberPosition: {zaber_position}")
        for axis in ['x', 'y', 'z']:
            val = getattr(zaber_position, axis)
            if val is not None:
                self._position[axis] = val

    def get_available_axes(self) -> list[str]:
        print("[Dummy] Reporting available axes: ['x', 'y', 'z']")
        return ['x', 'y', 'z']

    def close(self):
        print("[Dummy] Closing dummy connection")

    def shutdown(self):
        print("[Dummy] Shutting down dummy microscope")
        for key in self._led_states:
            self._led_states[key] = False

    # Simulated LED control attributes
    class DummyLED:
        def __init__(self, label):
            self.label = label
            self.state = False

        def on(self):
            self.state = True
            print(f"[Dummy] LED '{self.label}' turned ON")

        def off(self):
            self.state = False
            print(f"[Dummy] LED '{self.label}' turned OFF")

    # Dummy LED channels
    @property
    def led_white_below(self): return self.DummyLED("white_below")

    @property
    def led_blue_385_below(self): return self.DummyLED("blue_385_below")

    @property
    def led_red_625_below(self): return self.DummyLED("red_625_below")

    @property
    def led_white_top(self): return self.DummyLED("white_top")





if __name__ == "__main__":
    microscope = ZaberMicroscope()
