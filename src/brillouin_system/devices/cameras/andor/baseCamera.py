# baseCamera.py
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

import numpy as np

from brillouin_system.devices.cameras.andor.andor_dataclasses import AndorExposure, AndorCameraInfo


class BaseCamera(ABC):

    @abstractmethod
    def get_camera_info(self) -> dict:
        pass

    @abstractmethod
    def get_camera_info_dataclass(self) -> AndorCameraInfo:
        pass

    @abstractmethod
    def set_from_camera_info(self, info: AndorCameraInfo, do_set_temperature: bool = False):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def snap(self) -> tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def set_exposure_time(self, seconds: float):
        pass

    @abstractmethod
    def get_exposure_time(self) -> float:
        pass

    @abstractmethod
    def set_emccd_gain(self, gain: int):
        pass

    @abstractmethod
    def get_emccd_gain(self) -> int:
        pass

    @abstractmethod
    def get_preamp_gain(self) -> int:
        pass

    @abstractmethod
    def get_amp_mode(self) -> str:
        pass

    @abstractmethod
    def set_roi(self, x_start: int, x_end: int, y_start: int, y_end: int):
        pass

    @abstractmethod
    def get_roi(self) -> tuple[int, int, int, int]:
        pass

    @abstractmethod
    def set_binning(self, hbin: int, vbin: int):
        pass

    @abstractmethod
    def get_binning(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def get_frame_shape(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_verbose(self) -> bool:
        pass

    @abstractmethod
    def set_verbose(self, verbose: bool) -> None:
        pass

    @abstractmethod
    def open_shutter(self):
        pass

    @abstractmethod
    def close_shutter(self):
        pass

    # --- Newly added methods ---
    @abstractmethod
    def set_flip_image_horizontally(self, flip: bool):
        pass

    @abstractmethod
    def get_flip_image_horizontally(self) -> bool:
        pass

    @abstractmethod
    def set_fixed_pre_amp_mode(self, index: int):
        pass

    @abstractmethod
    def get_fixed_pre_amp_mode(self) -> int:
        pass

    @abstractmethod
    def get_pre_amp_mode(self) -> int:
        pass

    @abstractmethod
    def set_vss_index(self, index: int):
        pass

    @abstractmethod
    def get_vss_index(self) -> int:
        pass

    @abstractmethod
    def set_from_config_file(self, config) -> None:
        pass

    @abstractmethod
    def get_exposure_dataclass(self) -> AndorExposure:
        pass

    @abstractmethod
    def start_streaming(self, buffer_size: int = 200):
        pass

    @abstractmethod
    def stop_streaming(self):
        pass

    @abstractmethod
    def get_newest_streaming_image(self):
        """
        only when streaming
        Return the newest frame available (non-blocking).
        Returns None if no *new* frame since last call.
        """
        pass

    @abstractmethod
    def streaming(self) -> AbstractContextManager:
        """
        Return a context manager that starts streaming on enter
        and stops streaming on exit.
        """
        pass