from dataclasses import dataclass


@dataclass
class FlirCameraInfo:
    model: str
    serial: str
    sensor_size: tuple[int, int]
    roi: tuple[int, int, int, int]
    gain: float
    exposure: float
    pixel_format: str