from dataclasses import dataclass

import numpy as np


@dataclass
class AndorCameraSettings:
    """
    Container for configuration parameters used in scientific or industrial imaging systems.

    Attributes:
        name (str): Identifier or model name of the camera (e.g., "IxonUltra").

        exposure_time_s (float): Exposure time in seconds. Controls how long the sensor integrates light.

        emccd_gain (int): Electron Multiplying (EM) gain setting. Enhances signal in low-light conditions;
                          typically ranges from 0 to 300+.

        roi (tuple[int, int, int, int]): Region of interest as (x_start, x_end, y_start, y_end), defining
                                         the active sensor area used for acquisition.

        binning (tuple[int, int]): Pixel binning configuration as (horizontal_bin, vertical_bin). Higher
                                   binning reduces resolution but increases sensitivity and speed.

        preamp_gain (int | float): Pre-amplifier gain setting. Applied before digitization to adjust signal
                                  strength; value depends on camera capabilities.

        amp_mode (object): Encapsulates amplifier-related settings such as output amplifier index, horizontal
                           shift speed, and analog channel. Should be replaced with a well-defined class or
                           named tuple for clarity and type safety.
    """
    name: str
    exposure_time_s: float
    emccd_gain: int
    roi: tuple[int, int, int, int]
    binning: tuple[int, int]
    preamp_gain: int | float
    preamp_mode: str
