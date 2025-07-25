from dataclasses import dataclass


@dataclass
class AndorExposure:
    exposure_time_s: float
    emccd_gain: int


@dataclass
class AndorCameraInfo:
    model: str
    serial: str | int
    roi: tuple[int, int, int, int]
    binning: tuple[int, int]
    gain: int
    exposure: float
    amp_mode: str
    preamp_gain: float
    temperature: str | float
    flip_image_horizontally: bool
    advanced_gain_option: bool
    vss_speed: float | int