
import abc


class BaseDualCameras(abc.ABC):
    """
    Abstract base class for managing two synchronized cameras.
    Provides a common interface for real and dummy implementations.
    """



    @abc.abstractmethod
    def snap_once(self, timeout=5.0):
        """
        Capture one frame from each camera.
        Subclasses must implement how frames are obtained.
        Should return: (frame0, frame1)
        """
        pass

