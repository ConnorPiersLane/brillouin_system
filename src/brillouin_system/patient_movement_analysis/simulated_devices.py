"""
Simulated NI + Zaber eye lens with a moving simulated cornea.

Purpose: a FUNCTIONAL dummy mode for the patient-movement GUI and a test rig
for the cornea tracker. Unlike NIDummy (which lacks the background-acquisition
API), these devices implement the full API surface used by
find_reflection_realtime and CorneaTracker, and the DAQ signal is physically
coupled to the lens/cornea geometry:

    signal(t) = background + peak_v * exp(-(z_lens(t) - z_cornea(t))^2 / (2 sigma^2)) + noise

so the reflection finder actually finds the simulated cornea, and the tracker
actually tracks its motion (sinusoid + drift). All timestamps use
time.perf_counter(), like the real devices.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_position_log import ZaberPositionLog


# ----------------------------------------------------------------------
# Simulated cornea motion
# ----------------------------------------------------------------------

class SimulatedCornea:
    """z(t) = base + amplitude * sin(2*pi*f*(t-t0) ) + drift*(t-t0)  [um]"""

    def __init__(
        self,
        *,
        base_um: float = 9700.0,
        amplitude_um: float = 20.0,
        freq_hz: float = 0.3,
        drift_um_s: float = 0.5,
    ):
        self.base_um = float(base_um)
        self.amplitude_um = float(amplitude_um)
        self.freq_hz = float(freq_hz)
        self.drift_um_s = float(drift_um_s)
        self._t0 = time.perf_counter()

    def z_at(self, t: float) -> float:
        dt = t - self._t0
        return (self.base_um
                + self.amplitude_um * math.sin(2.0 * math.pi * self.freq_hz * dt)
                + self.drift_um_s * dt)

    def z_at_array(self, t: np.ndarray) -> np.ndarray:
        dt = np.asarray(t, dtype=np.float64) - self._t0
        return (self.base_um
                + self.amplitude_um * np.sin(2.0 * np.pi * self.freq_hz * dt)
                + self.drift_um_s * dt)


# ----------------------------------------------------------------------
# Simulated Zaber eye lens (piecewise-linear motion model)
# ----------------------------------------------------------------------

class SimZaberLens:
    """
    API-compatible stand-in for ZaberEyeLens (subset used by the finder,
    the tracker, and the backend). Motion is modeled as piecewise-constant
    velocity segments, so z_at(t) is exact for any past t — which the
    simulated NI uses to generate a physically consistent DAQ signal.
    """

    def __init__(self, *, start_um: float = 8000.0, move_speed_um_s: float = 5000.0,
                 read_latency_s: float = 0.012, position_noise_um: float = 0.2,
                 seed: Optional[int] = None):
        self._segments: list[tuple[float, float, float]] = [
            (time.perf_counter(), float(start_um), 0.0)]
        self._seg_lock = threading.Lock()
        self.move_speed_um_s = float(move_speed_um_s)
        self.read_latency_s = float(read_latency_s)
        self.position_noise_um = float(position_noise_um)
        self._rng = np.random.default_rng(seed)

        # guard / log state (mirrors real class)
        self._slew_guard_active = False
        self._log_active = False
        self._log_thread: Optional[threading.Thread] = None
        self._log_lock = threading.Lock()
        self._log_t: list[float] = []
        self._log_z: list[float] = []

    # ---------------- motion model ----------------

    def z_at(self, t: float) -> float:
        with self._seg_lock:
            seg = self._segments[0]
            for s in self._segments:
                if s[0] <= t:
                    seg = s
                else:
                    break
            t0, z0, v = seg
            return z0 + v * (t - t0)

    def z_at_array(self, t: np.ndarray) -> np.ndarray:
        return np.array([self.z_at(float(x)) for x in np.asarray(t)], dtype=np.float64)

    def _set_velocity(self, v: float) -> None:
        now = time.perf_counter()
        z = self.z_at(now)
        with self._seg_lock:
            self._segments.append((now, z, float(v)))
            if len(self._segments) > 10_000:
                self._segments = self._segments[-5000:]

    # ---------------- ZaberEyeLens API ----------------

    def home(self):
        self._set_velocity(0.0)

    def move_abs(self, position_um: float):
        now = time.perf_counter()
        delta = float(position_um) - self.z_at(now)
        if abs(delta) < 1e-9:
            return
        self._set_velocity(math.copysign(self.move_speed_um_s, delta))
        time.sleep(abs(delta) / self.move_speed_um_s)
        now2 = time.perf_counter()
        with self._seg_lock:
            self._segments.append((now2, float(position_um), 0.0))

    def move_rel(self, delta_um: float):
        self.move_abs(self.z_at(time.perf_counter()) + float(delta_um))

    def get_position(self) -> float:
        # Simulate the serial round-trip: the position is latched roughly
        # mid-transaction. The alpha=0.25 timestamp heuristic then leaves a
        # small residual latency bias (~ +6 um at 2000 um/s), comparable to
        # the ~+5 um measured on hardware — the tracker's up/down pair
        # averaging is expected to cancel it.
        time.sleep(0.5 * self.read_latency_s)
        z = self.z_at(time.perf_counter())
        time.sleep(0.5 * self.read_latency_s)
        return z + float(self._rng.normal(0.0, self.position_noise_um))

    def start_slewing(self, speed_um_per_s: float):
        self._set_velocity(float(speed_um_per_s))

    def stop_slewing(self):
        self._slew_guard_active = False
        self._set_velocity(0.0)

    def start_slewing_guarded(self, speed_um_per_s: float, max_distance_um: float):
        if max_distance_um <= 0:
            return
        start_pos = self.z_at(time.perf_counter())
        max_dist = abs(float(max_distance_um))
        self._slew_guard_active = True
        self._set_velocity(float(speed_um_per_s))

        def _guard_loop():
            try:
                while self._slew_guard_active:
                    travelled = abs(self.z_at(time.perf_counter()) - start_pos)
                    if travelled >= max_dist:
                        self._set_velocity(0.0)
                        break
                    time.sleep(0.03)
            finally:
                self._slew_guard_active = False

        threading.Thread(target=_guard_loop, daemon=True).start()

    # ---------------- position log (same behavior as real class) ----------------

    def start_position_log(self, *, poll_s: float = 0.016, alpha=0.25) -> None:
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be within [0, 1]")
        if self._log_active:
            raise RuntimeError("Position logging already running")

        with self._log_lock:
            self._log_t = []
            self._log_z = []
        self._log_active = True

        def _loop():
            next_t = time.perf_counter()
            try:
                while self._log_active:
                    now = time.perf_counter()
                    if now < next_t:
                        time.sleep(next_t - now)
                        continue
                    next_t += poll_s
                    t0 = time.perf_counter()
                    z = self.get_position()
                    t1 = time.perf_counter()
                    t = t0 + alpha * (t1 - t0)
                    with self._log_lock:
                        self._log_t.append(t)
                        self._log_z.append(z)
            finally:
                self._log_active = False

        th = threading.Thread(target=_loop, name="SimZaberPosLog", daemon=True)
        self._log_thread = th
        th.start()

    def stop_position_log(self, *, join_timeout_s: float = 2.0) -> ZaberPositionLog:
        if not self._log_active:
            return ZaberPositionLog(t_perf=np.empty(0), z_um=np.empty(0))
        self._log_active = False
        th = self._log_thread
        if th is not None:
            th.join(timeout=join_timeout_s)
        with self._log_lock:
            t = np.asarray(self._log_t, dtype=np.float64)
            z = np.asarray(self._log_z, dtype=np.float64)
        if t.size >= 2:
            keep = np.concatenate(([True], np.diff(t) > 0))
            t, z = t[keep], z[keep]
        return ZaberPositionLog(t_perf=t, z_um=z)

    def close(self):
        pass


# ----------------------------------------------------------------------
# Simulated NI DAQ
# ----------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SimReadResult:
    """Duck-typed NIReadResult (values / sample_rate_hz / t0_perf / ind0)."""
    values: np.ndarray
    sample_rate_hz: float
    t0_perf: float
    ind0: int

    def time_of(self, i) -> float:
        return self.t0_perf + (float(i) / float(self.sample_rate_hz))

    def timestamps_perf(self) -> np.ndarray:
        n = int(self.values.size)
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        return self.time_of(self.ind0) + np.arange(n, dtype=np.float64) / float(self.sample_rate_hz)


class SimNI:
    """
    API-compatible stand-in for NI6008 (subset used by the finder/tracker),
    generating the reflection signal from the simulated lens/cornea geometry.
    Peak width sigma default 3.8 um corresponds to the measured ~9 um FWHM.
    """

    def __init__(
        self,
        zaber: SimZaberLens,
        cornea: SimulatedCornea,
        *,
        sample_rate_hz: float = 1000.0,
        peak_v: float = 2.0,
        peak_sigma_um: float = 3.8,
        background_v: float = 0.002,
        noise_std_v: float = 0.003,
        seed: Optional[int] = None,
    ):
        self.zaber = zaber
        self.cornea = cornea
        self.sample_rate_hz = float(sample_rate_hz)
        self.peak_v = float(peak_v)
        self.peak_sigma_um = float(peak_sigma_um)
        self.background_v = float(background_v)
        self.noise_std_v = float(noise_std_v)
        self._rng = np.random.default_rng(seed)

        self._streaming = False
        self._t0 = 0.0          # perf time of FIFO sample index 0
        self._cursor = 0        # next unread FIFO index
        self._acq: Optional[dict] = None

    # ---------------- config ----------------

    def get_sample_rate_hz(self) -> float:
        return self.sample_rate_hz

    def set_sample_rate_hz(self, fs: float) -> None:
        if self._streaming:
            raise RuntimeError("Cannot change sample rate while streaming")
        self.sample_rate_hz = float(fs)

    # ---------------- lifecycle ----------------

    @contextmanager
    def streaming(self):
        if self._streaming:
            raise RuntimeError("Already streaming")
        self._streaming = True
        self._t0 = time.perf_counter()
        self._cursor = 0
        self._acq = None
        try:
            yield
        finally:
            self._streaming = False
            self._acq = None

    def _ensure_streaming(self):
        if not self._streaming:
            raise RuntimeError("Not streaming")

    # ---------------- signal generation ----------------

    def _gen(self, i0: int, i1: int) -> np.ndarray:
        """Generate FIFO samples for indices [i0, i1)."""
        n = i1 - i0
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        t = self._t0 + (np.arange(i0, i1, dtype=np.float64) / self.sample_rate_hz)
        dz = self.zaber.z_at_array(t) - self.cornea.z_at_array(t)
        signal = self.background_v + self.peak_v * np.exp(
            -(dz ** 2) / (2.0 * self.peak_sigma_um ** 2))
        return signal + self._rng.normal(0.0, self.noise_std_v, n)

    def _n_available(self) -> int:
        return int((time.perf_counter() - self._t0) * self.sample_rate_hz)

    # ---------------- foreground API ----------------

    def flush(self, **kw) -> int:
        self._ensure_streaming()
        n = self._n_available()
        discarded = n - self._cursor
        self._cursor = n
        return max(0, discarded)

    def read_block(self, n: int, *, timeout_s: float = 1.0) -> list[float]:
        self._ensure_streaming()
        n = int(n)
        if n <= 0:
            return []
        end_idx = self._cursor + n
        deadline = time.perf_counter() + timeout_s
        while self._n_available() < end_idx:
            if time.perf_counter() > deadline:
                break
            time.sleep(0.002)
        vals = self._gen(self._cursor, end_idx)
        self._cursor = end_idx
        return [float(x) for x in vals]

    # ---------------- background acquisition API ----------------

    def start_acquiring(self, *, max_sampling_time_s: float,
                        chunk_size: int = 1024, idle_sleep_s: float = 0.001) -> None:
        self._ensure_streaming()
        if self._acq is not None:
            raise RuntimeError("Already acquiring")
        start_idx = self._n_available()
        self._acq = {
            "start_idx": start_idx,
            "t0_perf": self._t0 + start_idx / self.sample_rate_hz,
            "max_samples": int(max_sampling_time_s * self.sample_rate_hz + chunk_size * 2),
            "values": np.empty(0, dtype=np.float64),  # cache for consistency
        }

    def _acq_extend(self) -> None:
        acq = self._acq
        have = acq["values"].size
        avail = min(self._n_available() - acq["start_idx"], acq["max_samples"])
        if avail > have:
            new = self._gen(acq["start_idx"] + have, acq["start_idx"] + avail)
            acq["values"] = np.concatenate([acq["values"], new])

    def get_new_block_result(self, last_idx: int, *, copy: bool = False) -> SimReadResult:
        self._ensure_streaming()
        acq = self._acq
        if acq is None:
            return SimReadResult(np.empty(0), self.sample_rate_hz, time.perf_counter(), int(last_idx))
        self._acq_extend()
        start = max(0, int(last_idx))
        vals = acq["values"]
        if start >= vals.size:
            return SimReadResult(np.empty(0), self.sample_rate_hz, acq["t0_perf"], start)
        out = vals[start:]
        return SimReadResult(out.copy() if copy else out,
                             self.sample_rate_hz, acq["t0_perf"], start)

    def get_acquiring_error(self) -> Optional[str]:
        return None

    def stop_acquiring(self, **kw) -> SimReadResult:
        acq = self._acq
        if acq is None:
            return SimReadResult(np.empty(0), self.sample_rate_hz, time.perf_counter(), 0)
        self._acq_extend()
        vals = acq["values"]
        # advance foreground cursor past the acquired region
        self._cursor = acq["start_idx"] + vals.size
        self._acq = None
        return SimReadResult(vals.copy(), self.sample_rate_hz, acq["t0_perf"], 0)
