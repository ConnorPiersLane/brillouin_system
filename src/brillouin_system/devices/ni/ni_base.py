from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class SupportsSampleRate(Protocol):
    sample_rate_hz: float


class NIBase(ABC):
    """
    Abstract base class for NI-like analog input devices.

    Matches NI6008-style API:

      - with ni.streaming():
      - ni.sample_rate_hz  (attribute/property)
      - ni.read_latest(timeout_s=...) -> float
      - ni.read_block(n, timeout_s=...) -> list[float]
      - ni.read_available_block() -> list[float]
      - ni.flush() -> int

    Notes:
      - NI6008 implements sample_rate_hz as an attribute; this base requires it too.
      - get_sample_rate() is kept for backward compatibility.
    """

    # ---- lifecycle ----

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

    # ---- sample rate ----

    @property
    @abstractmethod
    def get_sample_rate_hz(self) -> float:
        """Sample rate in Hz."""
        raise NotImplementedError


    # ---- reads ----

    @abstractmethod
    def read_latest(self, *, timeout_s: float = 0.05) -> float:
        """Return the most recent acquired sample."""
        raise NotImplementedError

    @abstractmethod
    def read_block(self, n: int, *, timeout_s: float = 1.0) -> list[float]:
        """
        Return the next n sequential samples (FIFO). May return fewer if timeout expires.
        """
        raise NotImplementedError

    @abstractmethod
    def read_available_block(self) -> list[float]:
        """Return all currently buffered samples (non-blocking), clearing the internal FIFO."""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> int:
        """Discard buffered samples. Returns the number of samples discarded."""
        raise NotImplementedError