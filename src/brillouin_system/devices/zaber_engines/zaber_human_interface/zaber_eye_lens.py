from dataclasses import dataclass

from zaber_motion import Library, Units
from zaber_motion.ascii import Connection


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
        self.axis = devices[0].get_axis(axis_index)

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


    def close(self):
        self.connection.close()


class ZaberEyeLensDummy:
    def __init__(self, port="COM5", axis_index=1):
        self.port = port
        self.axis_index = axis_index
        self._position = 100.0
        self.homed = False
        print(f"[ZaberEyeLensDummy] Initialized on port {port}, axis {axis_index}")
        self.home()

    def home(self):
        self._position = 100.0
        self.homed = True
        print("[ZaberEyeLensDummy] Homed. Position reset to 0.0 µm")

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



    def close(self):
        print(f"[ZaberEyeLensDummy] Shutdown: port {self.port}, axis {self.axis_index}")

if __name__ == "__main__":
    zaber_hi = ZaberEyeLens()