# baseFlirCamera.py

from abc import ABC, abstractmethod
import numpy as np
from brillouin_system.devices.cameras.flir.flir_config.flir_config import FLIRConfig
from brillouin_system.devices.cameras.flir.flir_dataclass import FlirCameraInfo


class BaseFLIRCamera(ABC):

    @abstractmethod
    def get_camera_info(self) -> dict:
        pass

    @abstractmethod
    def get_camera_info_dataclass(self) -> FlirCameraInfo:
        pass

    @abstractmethod
    def get_resolution(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def set_resolution(self, width: int, height: int):
        pass

    @abstractmethod
    def get_sensor_size(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def set_max_roi(self):
        pass

    @abstractmethod
    def set_roi_native(self, offset_x: int, offset_y: int, width: int, height: int):
        pass

    @abstractmethod
    def get_roi_native(self) -> tuple[int, int, int, int]:
        pass

    @abstractmethod
    def set_gain(self, value_dB: float):
        pass

    @abstractmethod
    def get_gain(self) -> float:
        pass

    @abstractmethod
    def min_max_gain(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def set_gamma(self, value: float):
        pass

    @abstractmethod
    def get_gamma(self) -> float | None:
        pass

    @abstractmethod
    def min_max_gamma(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def set_exposure_time(self, value: float):
        pass

    @abstractmethod
    def get_exposure_time(self) -> float:
        pass

    @abstractmethod
    def min_max_exposure_time(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def get_pixel_format(self) -> str:
        pass

    @abstractmethod
    def set_pixel_format(self, format_str: str):
        pass

    @abstractmethod
    def get_available_pixel_formats(self) -> list[str]:
        pass

    @abstractmethod
    def start_single_frame_mode(self):
        pass

    @abstractmethod
    def acquire_image(self, timeout: int = 1000) -> np.ndarray:
        pass

    @abstractmethod
    def start_software_stream(self):
        pass

    @abstractmethod
    def software_snap_while_stream(self, timeout: int = 1000) -> np.ndarray:
        pass

    @abstractmethod
    def end_software_stream(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass
