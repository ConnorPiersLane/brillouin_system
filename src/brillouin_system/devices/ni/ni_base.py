from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator, List


class NIBase(ABC):
    """
    Abstract base class for NI-like analog input devices.

    Expected API (matches your NI6008 + dummy patterns):
      - with ni.streaming():
      - ni.read_latest(timeout_s=...) -> float
      - ni.read_block(n, timeout_s=...) -> list[float]
      - ni.read_available_block() -> list[float]
      - ni.flush() -> int
      - ni.sample_rate_hz (attribute/property)
    """


    @contextmanager
    @abstractmethod
    def streaming(self) -> Iterator[None]:
        """
        Start continuous acquisition and stop/cleanup on exit.

        Usage:
            with ni.streaming():
                v = ni.read_latest()
        """
        yield  # pragma: no cover

    @abstractmethod
    def read_latest(self, *, timeout_s: float = 0.05) -> float:
        """Return the most recent acquired sample."""
        raise NotImplementedError

    @abstractmethod
    def read_block(self, n: int, *, timeout_s: float = 1.0) -> List[float]:
        """
        Return the next n samples (FIFO). Implementations may return fewer if timeout expires.
        """
        raise NotImplementedError

    @abstractmethod
    def read_available_block(self) -> List[float]:
        """Return all currently buffered samples (non-blocking), clearing the internal FIFO."""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> int:
        """Discard buffered samples. Returns the number of samples discarded."""
        raise NotImplementedError