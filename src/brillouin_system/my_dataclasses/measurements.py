from dataclasses import dataclass


@dataclass
class MeasurementSettings:
    n_measurements: int
    name: str = 'Unnamed'
    power_mW: float = 0.0
    move_axes: str = ''
    move_x_rel_um: float = 0.0
    move_y_rel_um: float = 0.0
    move_z_rel_um: float = 0.0



