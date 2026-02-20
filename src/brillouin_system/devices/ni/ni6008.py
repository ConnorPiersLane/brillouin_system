from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Optional, Literal, Tuple

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration


@dataclass(frozen=True)
class StepEvent:
    t: float                 # wall-clock time (time.time()) when detected
    value: float             # sample value at detection
    baseline: Optional[float] = None
    delta: Optional[float] = None


class NI6008:
    """
    Simple NI USB-6008 analog-input reader (no threads).

    Typical PDA10A wiring:
      - BNC center (+) -> AI0 (AI0+)
      - BNC shell  (-) -> GND (AIGND)
      - Use RSE terminal configuration

    Usage:
        ni = NI6008(device="Dev1", ai_channel="ai0", sample_rate_hz=1000)
        with ni.streaming():
            v = ni.read_value()
            bg_mean, bg_std = ni.get_background_value(n_samples=500)
            evt = ni.wait_for_step(threshold=0.2, mode="delta", direction="rising")
    """

    def __init__(
        self,
        device: str = "Dev1",
        ai_channel: str = "ai0",
        *,
        terminal_config: TerminalConfiguration = TerminalConfiguration.RSE,
        min_val: float = -10.0,
        max_val: float = 10.0,
        sample_rate_hz: float = 1000.0,
    ):
        self.device = device
        self.ai_channel = ai_channel
        self.terminal_config = terminal_config
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.sample_rate_hz = float(sample_rate_hz)

        self._task: Optional[nidaqmx.Task] = None

    @property
    def physical_channel(self) -> str:
        return f"{self.device}/{self.ai_channel}"

    @contextmanager
    def streaming(self):
        """Create/start a continuous task; close it on exit."""
        if self._task is not None:
            raise RuntimeError("NI6008 is already streaming")

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
                samps_per_chan=10,
            )

            task.start()
            self._task = task
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

    def read_value(self, *, timeout_s: float = 1.0) -> float:
        """Read one sample (volts). Must be inside `with streaming():`."""
        if self._task is None:
            raise RuntimeError("Not streaming. Use `with ni.streaming():`.")
        v = self._task.read(number_of_samples_per_channel=1, timeout=float(timeout_s))
        return v[0]

    def read_block(self, n_samples: int, *, timeout_s: Optional[float] = None) -> list[float]:
        """Read N samples as a list[float]."""
        if self._task is None:
            raise RuntimeError("Not streaming. Use `with ni.streaming():`.")
        n = int(n_samples)
        if n <= 0:
            return []

        if timeout_s is None:
            timeout_s = max(1.0, n / self.sample_rate_hz + 0.5)

        data = self._task.read(number_of_samples_per_channel=n, timeout=float(timeout_s))

        return [float(x) for x in data]

    def flush(self, *, max_reads: int = 50) -> int:
        """
        Drain unread samples from the DAQmx buffer.

        seconds > 0: also discard ~seconds worth of *new* samples after the buffer is empty,
        so the next read reflects the current moment.
        Returns: total samples discarded.
        """
        if self._task is None:
            raise RuntimeError("Not streaming. Use `with ni.streaming():`.")

        total = 0

        # 1) Drain what's already buffered (available immediately)
        for _ in range(max_reads):
            avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
            if avail <= 0:
                break
            data = self._task.read(number_of_samples_per_channel=avail, timeout=0.0)
            total += len(data)

        return total


if __name__ == "__main__":
    ni = NI6008(device="Dev1", ai_channel="ai0", sample_rate_hz=1000)

    with ni.streaming():
        vals = ni.read_block(100)
        print(f"Read {len(vals)} samples:")
        print(vals)
        print(ni.read_value())
        print(ni.flush())
