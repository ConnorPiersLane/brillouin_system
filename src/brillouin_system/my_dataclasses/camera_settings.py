from dataclasses import dataclass


@dataclass
class CameraSettings:
    name: str
    exposure_time_s: float
    gain: int
    roi: tuple[int, int, int, int]
    binning: tuple[int, int]
    amp_mode: tuple