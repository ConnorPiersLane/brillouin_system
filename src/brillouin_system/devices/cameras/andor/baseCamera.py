from abc import ABC, abstractmethod
import numpy as np

class BaseCamera(ABC):

    @abstractmethod
    def get_name(self) -> str:
        """ Get the camera name or type """
        pass

    @abstractmethod
    def snap(self) -> np.ndarray:
        """Acquire a single image and return it as a NumPy array."""
        pass

    @abstractmethod
    def set_exposure_time(self, seconds: float):
        """Set the camera exposure time in seconds."""
        pass

    @abstractmethod
    def get_exposure_time(self) -> float:
        """Get the current exposure time in seconds."""
        pass

    @abstractmethod
    def set_gain(self, gain: float | int):
        """Set the camera gain."""
        pass

    @abstractmethod
    def get_gain(self) -> float | int:
        """Get the current camera gain."""
        pass

    @abstractmethod
    def set_roi(self, x_start: int, x_end: int, y_start: int, y_end: int):
        """Set the region of interest."""
        pass

    @abstractmethod
    def get_roi(self) -> tuple[int, int, int, int]:
        """Get the current region of interest as (x_start, x_end, y_start, y_end)."""
        pass

    @abstractmethod
    def set_binning(self, hbin: int, vbin: int):
        """Set horizontal and vertical binning."""
        pass

    @abstractmethod
    def get_binning(self) -> tuple[int, int]:
        """Get the current horizontal and vertical binning."""
        pass

    @abstractmethod
    def get_frame_shape(self) -> tuple[int, int]:
        """Return the expected shape (height, width) of the next image."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Return True if the camera is currently open and initialized."""
        pass

    @abstractmethod
    def close(self):
        """Close the camera and release any resources."""
        pass

    @abstractmethod
    def get_verbose(self) -> bool:
        """ If True, the camera prints extra information to print() """
        pass

    @abstractmethod
    def set_verbose(self, verbose: bool) -> None:
        """ Set if the camera is verbose or not """
        pass

    @abstractmethod
    def open_shutter(self):
        pass

    @abstractmethod
    def close_shutter(self):
        pass
