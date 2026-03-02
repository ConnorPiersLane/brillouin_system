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
    Minimal NI USB-6008 analog input helper.

    - read_latest() -> most recent acquired sample
    - read_block(n) -> next n samples (FIFO)
    - flush()       -> discard buffered samples (so next read starts "now")
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
    ):
        self.device = device
        self.ai_channel = ai_channel
        self.sample_rate_hz = float(sample_rate_hz)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.terminal_config = terminal_config

        self._task: Optional[nidaqmx.Task] = None
        self._reader: Optional[AnalogSingleChannelReader] = None

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
                samps_per_chan=1000,
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

    def _ensure_streaming(self):
        if self._task is None or self._reader is None:
            raise RuntimeError("Not streaming")

    # ---------- latest ----------

    def read_latest(self, *, timeout_s: float = 0.05) -> float:
        self._ensure_streaming()

        self._task.in_stream.relative_to = ReadRelativeTo.MOST_RECENT_SAMPLE
        self._task.in_stream.offset = 0

        return float(self._reader.read_one_sample(timeout=float(timeout_s)))

    # ---------- FIFO block ----------

    def read_block(self, n: int, *, timeout_s: float = 1.0) -> list[float]:
        self._ensure_streaming()

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        data = self._task.read(
            number_of_samples_per_channel=int(n),
            timeout=float(timeout_s),
        )
        return [float(x) for x in data]

    # ---------- flush ----------
    def read_available_block(self) -> list[float]:
        self._ensure_streaming()

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
        if avail <= 0:
            return []

        data = self._task.read(
            number_of_samples_per_channel=avail,
            timeout=0.0,
        )

        return [float(x) for x in data]

    def flush(self) -> int:
        """
        Discard all currently buffered samples.
        Returns number of samples discarded.
        """
        self._ensure_streaming()

        total = 0
        while True:
            avail = int(self._task.in_stream.avail_samp_per_chan)
            if avail <= 0:
                break

            _ = self._task.read(
                number_of_samples_per_channel=avail,
                timeout=0.1
            )
            total += avail

        return total

if __name__ == "__main__":
    ni = NI6008(device="Dev1", ai_channel="ai0", sample_rate_hz=1000)

    with ni.streaming():
        time.sleep(0.2)
        print(ni.flush())
        print(np.std(ni.read_block(100)))
        print(ni.flush())
        t1 = time.monotonic()
        r = ni.read_latest()
        t2 = time.monotonic()
        print(f"time {t2 - t1}")
        t1 = time.monotonic()
        r = ni.read_block(1)
        t2 = time.monotonic()
        print(f"time {t2-t1}")
        print(ni.read_latest())
        print(ni.flush())
        print(ni.flush())
        r1 = ni.read_block(1000)
        r2 = ni.read_block(1000)
        print(r1+r2)