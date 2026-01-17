from dataclasses import dataclass
import re


@dataclass
class AndorExposure:
    exposure_time_s: float
    emccd_gain: int

def andor_exposures_equal(
    a: AndorExposure,
    b: AndorExposure,
    *,
    exposure_time_digits: int = 3,
) -> bool:
    """
    Compare two AndorExposure objects.

    - exposure_time_s is compared rounded to `exposure_time_digits`
    - emccd_gain must match exactly
    """
    if a.emccd_gain != b.emccd_gain:
        return False

    return round(a.exposure_time_s, exposure_time_digits) == round(
        b.exposure_time_s, exposure_time_digits
    )



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
    fixed_pre_amp_mode_index: int | None = None

