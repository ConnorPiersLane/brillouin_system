from __future__ import annotations

from contextlib import contextmanager
import random
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StepEvent:
    t: float
    value: float
    baseline: Optional[float] = None
    delta: Optional[float] = None


class NIDummy:
    """
    Dummy NI6008 replacement (no nidaqmx required).

    Same API:
      - with ni.streaming():
      - ni.read_value()
      - ni.read_block(n)
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float = 1000.0,
        noise_std: float = 0.01,
        baseline: float = 0.0,
        step_amplitude: float = 0.2,
        step_probability_per_sample: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.sample_rate_hz = float(sample_rate_hz)
        self.noise_std = float(noise_std)
        self.baseline = float(baseline)
        self.step_amplitude = float(step_amplitude)
        self.step_probability_per_sample = float(step_probability_per_sample)

        self._streaming = False
        self._current_level = self.baseline
        self._rng = random.Random(seed)

    @contextmanager
    def streaming(self):
        if self._streaming:
            raise RuntimeError("NI6008 dummy is already streaming")
        self._streaming = True
        try:
            yield
        finally:
            self._streaming = False

    def _next_sample(self) -> float:
        # optional random step events
        if self._rng.random() < self.step_probability_per_sample:
            self._current_level += self._rng.choice([-1.0, 1.0]) * self.step_amplitude
        return self._current_level + self._rng.gauss(0.0, self.noise_std)

    def read_value(self, *, timeout_s: float = 1.0) -> float:
        if not self._streaming:
            raise RuntimeError("Not streaming. Use `with ni.streaming():`.")
        time.sleep(1.0 / self.sample_rate_hz)
        return float(self._next_sample())

    def read_block(self, n_samples: int, *, timeout_s: Optional[float] = None) -> list[float]:
        if not self._streaming:
            raise RuntimeError("Not streaming. Use `with ni.streaming():`.")

        n = int(n_samples)
        if n <= 0:
            return []

        dt = 1.0 / self.sample_rate_hz
        out: list[float] = []
        for _ in range(n):
            out.append(float(self._next_sample()))
            time.sleep(dt)
        return out


if __name__ == "__main__":
    ni = NIDummy(sample_rate_hz=1000, noise_std=0.02, step_probability_per_sample=1e-3, seed=0)
    with ni.streaming():
        vals = ni.read_block(10)
        print(vals)
        print("single:", ni.read_value())