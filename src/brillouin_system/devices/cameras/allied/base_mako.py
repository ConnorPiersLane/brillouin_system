from abc import ABC, abstractmethod

class BaseMakoCamera(ABC):
    @abstractmethod
    def set_exposure(self, exposure_us: float):
        """Set the camera exposure time in microseconds."""
        pass

    @abstractmethod
    def get_exposure(self) -> float:
        """Get the current exposure time in microseconds."""
        pass

    @abstractmethod
    def set_gain(self, gain_db: float):
        """Set the camera gain in decibels."""
        pass

    @abstractmethod
    def get_gain(self) -> float:
        """Get the current gain in decibels."""
        pass

    @abstractmethod
    def set_roi(self, offset_x: int, offset_y: int, width: int, height: int):
        """Set the Region of Interest."""
        pass

    @abstractmethod
    def get_roi(self) -> dict:
        """Return the Region of Interest as a dict with keys OffsetX, OffsetY, Width, Height."""
        pass

    @abstractmethod
    def snap(self):
        """Capture a single frame."""
        pass

    @abstractmethod
    def start_stream(self, frame_callback, buffer_count=5):
        """Start streaming frames, using the given callback."""
        pass

    @abstractmethod
    def stop_stream(self):
        """Stop streaming."""
        pass

    @abstractmethod
    def close(self):
        """Clean up and release the camera."""
        pass
