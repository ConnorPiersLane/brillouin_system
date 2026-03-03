from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

import nidaqmx
import numpy as np
from nidaqmx.constants import (
    AcquisitionType,
    TerminalConfiguration,
    ReadRelativeTo,
)
from nidaqmx.stream_readers import AnalogSingleChannelReader

from brillouin_system.devices.ni.ni_base import NIBase


class NI6008(NIBase):
    """
    Minimal NI USB-6008 analog input helper (optimized).

    Key performance changes vs list-based reads:
      - Uses AnalogSingleChannelReader.read_many_sample(...) into a reusable NumPy buffer
      - Avoids Python list creation and per-element float conversion
      - Returns np.ndarray views (zero-copy slices)

    API:
      - read_latest() -> float
      - read_block(n) -> np.ndarray shape (n,)
      - read_available_block() -> np.ndarray shape (k,) (k may be 0)
      - flush() -> int (samples discarded)
    """

    def __init__(
        self,
        device: str = "Dev1",
        ai_channel: str = "ai0",
        *,
        sample_rate_hz: float = 1000.0,
        min_val: float = -10.0,
        max_val: float = 10.0,
        terminal_config: TerminalConfiguration = TerminalConfiguration.RSE,
        samps_per_chan_buffer: int = 1000,
    ):
        self.device = device
        self.ai_channel = ai_channel
        self.sample_rate_hz = float(sample_rate_hz)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.terminal_config = terminal_config
        self.samps_per_chan_buffer = int(samps_per_chan_buffer)

        self._task: Optional[nidaqmx.Task] = None
        self._reader: Optional[AnalogSingleChannelReader] = None

        # Reusable NumPy buffer to avoid per-call allocations
        self._buf: Optional[np.ndarray] = None
        self._buf_capacity: int = 0

        # Optional tiny empty array to return without allocating each time
        self._empty = np.empty(0, dtype=np.float64)

    @property
    def physical_channel(self) -> str:
        return f"{self.device}/{self.ai_channel}"

    # ---------- lifecycle ----------

    @contextmanager
    def streaming(self):
        if self._task is not None:
            raise RuntimeError("Already streaming")

        task = nidaqmx.Task()
        try:
            task.ai_channels.add_ai_voltage_chan(
                self.physical_channel,
                min_val=self.min_val,
                max_val=self.max_val,
                terminal_config=self.terminal_config,
            )

            task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate_hz,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samps_per_chan_buffer,
            )

            task.start()

            self._task = task
            self._reader = AnalogSingleChannelReader(task.in_stream)

            yield
        finally:
            try:
                task.stop()
            except Exception:
                pass
            try:
                task.close()
            except Exception:
                pass

            self._task = None
            self._reader = None

    def _ensure_streaming(self) -> None:
        if self._task is None or self._reader is None:
            raise RuntimeError("Not streaming")

    def _ensure_buffer(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return self._empty
        if self._buf is None or self._buf_capacity < n:
            self._buf = np.empty(n, dtype=np.float64)
            self._buf_capacity = n
        return self._buf

    def get_sample_rate(self) -> float:
        return self.sample_rate_hz

    # ---------- latest ----------

    def read_latest(self, *, timeout_s: float = 0.05) -> float:
        """Read the most recent acquired sample (single value)."""
        self._ensure_streaming()
        assert self._task is not None and self._reader is not None

        self._task.in_stream.relative_to = ReadRelativeTo.MOST_RECENT_SAMPLE
        self._task.in_stream.offset = 0

        return float(self._reader.read_one_sample(timeout=float(timeout_s)))

    # ---------- FIFO block ----------

    def read_block(self, n: int, *, timeout_s: float = 1.0) -> np.ndarray:
        """Read next n samples from FIFO, returned as np.ndarray (view)."""
        self._ensure_streaming()
        assert self._task is not None and self._reader is not None

        n = int(n)
        if n <= 0:
            return self._empty

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        buf = self._ensure_buffer(n)

        # Read directly into NumPy buffer (no Python list allocations)
        self._reader.read_many_sample(
            data=buf,
            number_of_samples_per_channel=n,
            timeout=float(timeout_s),
        )

        return buf[:n]

    def read_available_block(self, *, timeout_s: float = 0.01) -> np.ndarray:
        """Read all currently available samples from FIFO (may be empty)."""
        self._ensure_streaming()
        assert self._task is not None and self._reader is not None

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
        if avail <= 0:
            return self._empty

        buf = self._ensure_buffer(avail)

        self._reader.read_many_sample(
            data=buf,
            number_of_samples_per_channel=avail,
            timeout=float(timeout_s),
        )

        return buf[:avail]

    # ---------- flush ----------

    def flush(self) -> int:
        """
        Discard all currently buffered samples.
        Returns number of samples discarded.
        """
        self._ensure_streaming()
        assert self._task is not None and self._reader is not None

        total = 0
        while True:
            avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
            if avail <= 0:
                break

            buf = self._ensure_buffer(avail)
            # timeout=0.0 to avoid waiting; just drain what exists
            self._reader.read_many_sample(
                data=buf,
                number_of_samples_per_channel=avail,
                timeout=0.0,
            )
            total += avail

        return total


if __name__ == "__main__":
    ni = NI6008(device="Dev1", ai_channel="ai0", sample_rate_hz=1000)

    with ni.streaming():
        time.sleep(0.2)

        print("flushed:", ni.flush())
        xs = ni.read_block(100)
        print("std(100):", float(np.std(xs)))
        print("flushed:", ni.flush())

        t1 = time.monotonic()
        _ = ni.read_latest()
        t2 = time.monotonic()
        print(f"read_latest time: {t2 - t1:.6f} s")

        t1 = time.monotonic()
        _ = ni.read_block(1)
        t2 = time.monotonic()
        print(f"read_block(1) time: {t2 - t1:.6f} s")

        print("latest:", ni.read_latest())
        print("flushed:", ni.flush())
        print("flushed:", ni.flush())

        r1 = ni.read_block(1000)
        print("read_block(1000) shape:", r1.shape)