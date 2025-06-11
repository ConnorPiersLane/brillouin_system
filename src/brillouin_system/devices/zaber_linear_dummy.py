from brillouin_system.my_dataclasses.zaber_position import ZaberPosition


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

    def set_speed(self, which_axis: str, speed_mm_per_s: float):
        self._check_axis(which_axis)
        self.speed_mm_per_s = speed_mm_per_s
        print(f"[ZaberDummy] Speed for {which_axis.upper()} set to {speed_mm_per_s:.2f} mm/s")

    def set_acceleration(self, which_axis: str, accel_native_units: int):
        self._check_axis(which_axis)
        self.accel_native_units = accel_native_units
        print(f"[ZaberDummy] Acceleration for {which_axis.upper()} set to {accel_native_units}")

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
