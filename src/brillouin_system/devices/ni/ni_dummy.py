from __future__ import annotations

import time
import math
import random
from collections import deque
from contextlib import contextmanager
from typing import Deque, List, Optional

from brillouin_system.devices.ni.ni_base import NIBase


class NIDummy(NIBase):
    """
    Drop-in dummy replacement for NI6008.

    API-compatible with:
      - streaming()
      - read_latest()
      - read_block(n)
      - read_available_block()
      - flush()

    Extras:
      - gain (e.g. 20x larger values)
      - deterministic timing
      - never returns empty when it shouldn't
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float = 1000.0,
        baseline_v: float = 0.1,
        noise_std_v: float = 0.01,
        signal_amplitude_v: float = 0.5,
        signal_freq_hz: float = 2.0,
        gain: float = 1.0,              # <<< SET THIS TO 20.0 WHEN NEEDED
        max_buffer_seconds: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.sample_rate_hz = float(sample_rate_hz)
        self.baseline_v = float(baseline_v)
        self.noise_std_v = float(noise_std_v)
        self.signal_amplitude_v = float(signal_amplitude_v)
        self.signal_freq_hz = float(signal_freq_hz)
        self.gain = float(gain)

        self._rng = random.Random(seed)

        self._max_buffer_len = int(max_buffer_seconds * self.sample_rate_hz)
        self._buf: Deque[float] = deque()

        self._streaming = False
        self._t0 = 0.0
        self._last_index = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def streaming(self):
        if self._streaming:
            raise RuntimeError("Already streaming")

        self._streaming = True
        self._t0 = time.monotonic()
        self._last_index = 0
        self._buf.clear()

        try:
            yield
        finally:
            self._streaming = False
            self._buf.clear()

    def get_sample_rate(self) -> float:
        return self.sample_rate_hz

    def _ensure_streaming(self):
        if not self._streaming:
            raise RuntimeError("Not streaming")

    # ------------------------------------------------------------------
    # signal generation
    # ------------------------------------------------------------------

    def _sample_value(self, index: int) -> float:
        if index % 100 == 0:
            return 5.0

        # small deterministic variation, always < 1
        return 0.2 + 0.3 * math.sin(2 * math.pi * index / 25)

    def _update_buffer(self):
        self._ensure_streaming()

        now = time.monotonic()
        elapsed = max(0.0, now - self._t0)
        target_index = int(elapsed * self.sample_rate_hz)

        while self._last_index < target_index:
            self._buf.append(self._sample_value(self._last_index))
            self._last_index += 1

        # cap buffer size
        while len(self._buf) > self._max_buffer_len:
            self._buf.popleft()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def read_latest(self, *, timeout_s: float = 0.05) -> float:
        self._ensure_streaming()

        deadline = time.monotonic() + timeout_s
        while True:
            self._update_buffer()
            if self._buf:
                return self._buf[-1]

            if time.monotonic() >= deadline:
                # force-generate one sample
                v = self._sample_value(self._last_index)
                self._last_index += 1
                self._buf.append(v)
                return v

            time.sleep(0.001)

    def read_block(self, n: int, *, timeout_s: float = 1.0) -> List[float]:
        self._ensure_streaming()
        n = int(n)
        if n <= 0:
            return []

        deadline = time.monotonic() + timeout_s
        out: List[float] = []

        while len(out) < n:
            self._update_buffer()

            while self._buf and len(out) < n:
                out.append(self._buf.popleft())

            if len(out) >= n or time.monotonic() >= deadline:
                break

            time.sleep(0.001)

        return out

    def read_available_block(self) -> List[float]:
        self._ensure_streaming()
        self._update_buffer()

        if not self._buf:
            return []

        out = list(self._buf)
        self._buf.clear()
        return out

    def flush(self) -> int:
        self._ensure_streaming()
        self._update_buffer()
        n = len(self._buf)
        self._buf.clear()
        return n


# ----------------------------------------------------------------------
# quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ni = NIDummy(
        sample_rate_hz=1000,
        gain=20.0,          # <<< 20× larger values
        seed=0,
    )

    with ni.streaming():
        time.sleep(0.2)
        print("flush:", ni.flush())
        print("block std:", max(ni.read_block(100)))
        print("latest:", ni.read_latest())
        print("available:", len(ni.read_available_block()))