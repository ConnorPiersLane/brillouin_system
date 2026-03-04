from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, ReadRelativeTo, TerminalConfiguration
from nidaqmx.stream_readers import AnalogSingleChannelReader

from brillouin_system.devices.ni.ni_base import NIBase


# ----------------------------
# Results / internal state
# ----------------------------

@dataclass(frozen=True)
class AcquisitionResult:
    """Returned by stop_acquiring()."""
    values: np.ndarray          # shape (n,), float64
    sample_rate_hz: float       # DAQ sample rate
    t0_wall: float              # wall-clock time.time() when acquisition started

    def timestamps(self) -> np.ndarray:
        """
        Reconstruct timestamps on demand (no need to store them per-sample).
        t[i] = t0_wall + i / sample_rate_hz

        NOTE:
        This uses an ideal sample clock from t0_wall. It is consistent *relative*
        to itself, but absolute alignment to the ADC's first-sample wall time may
        be offset by buffering/OS scheduling.
        """
        n = self.values.size
        return self.t0_wall + (np.arange(n, dtype=np.float64) / self.sample_rate_hz)


@dataclass
class _AcqState:
    stop_evt: threading.Event
    thread: threading.Thread
    lock: threading.Lock
    running: bool

    # storage
    values: np.ndarray
    capacity: int
    write_idx: int

    # metadata
    t0_wall: float
    sample_rate_hz: float
    max_samples: Optional[int]


# ----------------------------
# NI6008 device
# ----------------------------

class NI6008(NIBase):
    """
    Minimal NI USB-6008 analog input helper (optimized).

    Foreground reads (read_latest/read_block/read_available_block/flush) are allowed
    ONLY when background acquisition is NOT running.

    Background acquisition design:
      - Dedicated daemon thread continuously drains the DAQ FIFO in blocks.
      - Stores ONLY values (timestamps are reconstructed from t0 + i/fs).
      - Preallocates if duration_s or max_samples provided; otherwise grows by doubling.
      - Never intentionally drops frames: if max_samples is a hard limit and buffer fills,
        acquisition stops instead of overwriting/dropping.
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
        samps_per_chan_buffer: int = 10_000,
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

        # Foreground reusable buffer (ONLY used in foreground methods; never by producer)
        self._fg_buf: Optional[np.ndarray] = None
        self._fg_buf_capacity: int = 0
        self._empty = np.empty(0, dtype=np.float64)

        # Background acquisition state (None if not acquiring)
        self._acq: Optional[_AcqState] = None

    @property
    def physical_channel(self) -> str:
        return f"{self.device}/{self.ai_channel}"

    # ----------------------------
    # lifecycle
    # ----------------------------

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
            acq = self._acq
            if acq is not None:
                try:
                    acq.stop_evt.set()
                except Exception:
                    pass

                # Wait for producer to stop before closing task
                acq.thread.join(timeout=2.0)
                if acq.thread.is_alive():
                    raise RuntimeError("NI6008 acquisition thread did not stop during streaming() cleanup")

                self._acq = None

            # Now it's safe to stop/close the DAQmx task
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
            raise RuntimeError("Not streaming (use `with ni.streaming(): ...`)")

    def _ensure_not_acquiring(self) -> None:
        if self._acq is not None and self._acq.running:
            raise RuntimeError("Foreground reads are disabled while background acquisition is running")

    def _ensure_fg_buffer(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return self._empty
        if self._fg_buf is None or self._fg_buf_capacity < n:
            self._fg_buf = np.empty(n, dtype=np.float64)
            self._fg_buf_capacity = n
        return self._fg_buf

    # ----------------------------
    # background acquisition
    # ----------------------------

    def start_acquiring(
        self,
        *,
        duration_s: float | None = None,
        max_samples: int | None = None,
        chunk_size: int = 2048,
        poll_timeout_s: float = 0.1,
        initial_capacity_s: float = 5.0,
    ) -> None:
        """
        Start continuous background acquisition into a memory buffer.

        Foreground reads are not allowed while acquisition is running.

        Args:
            duration_s:
                If provided, preallocates exactly ceil(fs * duration_s) samples.
            max_samples:
                Hard cap in samples. If reached, acquisition stops (does not drop/overwrite).
            chunk_size:
                Max samples to read per loop iteration.
            poll_timeout_s:
                Timeout for DAQ read calls; bounds blocking so stop_acquiring responds quickly.
            initial_capacity_s:
                When duration_s/max_samples not provided, start buffer around this many seconds.
        """
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if poll_timeout_s is not None and float(poll_timeout_s) < 0:
            raise ValueError("poll_timeout_s must be >= 0")

        if self._acq is not None and self._acq.running:
            raise RuntimeError("Acquisition already running")

        self._ensure_streaming()
        assert self._task is not None and self._reader is not None

        fs = float(self.sample_rate_hz)

        if duration_s is not None:
            if float(duration_s) <= 0:
                raise ValueError("duration_s must be > 0")
            cap = int(np.ceil(fs * float(duration_s)))
            cap = max(1, cap)
            max_samples_eff = cap if max_samples is None else min(int(max_samples), cap)
        else:
            max_samples_eff = None if max_samples is None else int(max_samples)
            if max_samples_eff is not None and max_samples_eff <= 0:
                raise ValueError("max_samples must be > 0")

            cap = max(int(fs * float(initial_capacity_s)), chunk_size * 4, 4096)
            if max_samples_eff is not None:
                cap = min(cap, max_samples_eff)

        values = np.empty(cap, dtype=np.float64)

        stop_evt = threading.Event()
        lock = threading.Lock()
        t0_wall = time.time()

        # Create state first so producer can reference it safely
        self._acq = _AcqState(
            stop_evt=stop_evt,
            thread=threading.Thread(),  # placeholder, replaced below
            lock=lock,
            running=True,
            values=values,
            capacity=cap,
            write_idx=0,
            t0_wall=t0_wall,
            sample_rate_hz=fs,
            max_samples=max_samples_eff,
        )

        def _grow_if_needed(acq: _AcqState, extra: int) -> None:
            needed = acq.write_idx + extra
            if needed <= acq.capacity:
                return

            if acq.max_samples is not None and needed > acq.max_samples:
                # Can't grow beyond hard cap.
                return

            new_cap = max(acq.capacity * 2, needed)
            if acq.max_samples is not None:
                new_cap = min(new_cap, acq.max_samples)

            new_values = np.empty(new_cap, dtype=np.float64)
            new_values[: acq.write_idx] = acq.values[: acq.write_idx]
            acq.values = new_values
            acq.capacity = new_cap

        def _producer() -> None:
            # Producer-local buffer: NEVER shared with foreground code
            tmp = np.empty(chunk_size, dtype=np.float64)

            # Tune this: small blocking read when FIFO is empty.
            # 8–64 is usually fine. At 1000 Hz, 32 samples ~= 32 ms.
            min_block = min(32, chunk_size)

            try:
                if self._task is None or self._reader is None:
                    return

                self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
                self._task.in_stream.offset = 0

                while not stop_evt.is_set():
                    if self._task is None or self._reader is None:
                        break

                    avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))

                    if avail > 0:
                        # Drain immediately (non-blocking)
                        want = min(avail, chunk_size)
                        timeout = 0.0
                    else:
                        # FIFO empty: block briefly for a small chunk so we actually keep acquiring
                        want = min_block
                        timeout = float(poll_timeout_s)

                        # If user set poll_timeout_s=0, don't busy-spin
                        if timeout == 0.0:
                            time.sleep(0.001)
                            continue

                    view = tmp[:want]
                    try:
                        n_read = self._reader.read_many_sample(
                            data=view,
                            number_of_samples_per_channel=want,
                            timeout=timeout,
                        )

                    except nidaqmx.errors.DaqError as e:
                        TIMEOUT = -200284  # DAQmx timeout

                        code = getattr(e, "error_code", None)
                        if hasattr(code, "value"):  # sometimes it's an enum-like object
                            code = code.value

                        if code == TIMEOUT:
                            continue  # benign timeout, keep acquiring

                        # real DAQ error -> stop
                        stop_evt.set()
                        break

                    except Exception:
                        stop_evt.set()
                        break

                    if n_read is None:
                        n_read = want
                    n_read = int(n_read)
                    if n_read <= 0:
                        continue

                    with lock:
                        acq = self._acq
                        if acq is None:
                            break

                        if acq.max_samples is not None and acq.write_idx >= acq.max_samples:
                            stop_evt.set()
                            break

                        _grow_if_needed(acq, n_read)

                        room = acq.capacity - acq.write_idx
                        take = n_read if n_read <= room else room

                        # Copy from the actual read view
                        acq.values[acq.write_idx: acq.write_idx + take] = view[:take]
                        acq.write_idx += take

                        if take < n_read and acq.max_samples is not None:
                            stop_evt.set()
                            break
            finally:
                with lock:
                    acq = self._acq
                    if acq is not None:
                        acq.running = False

        th = threading.Thread(target=_producer, name="NI6008Acq", daemon=True)
        self._acq.thread = th
        th.start()

    def stop_acquiring(self, *, join_timeout_s: float | None = 2.0) -> AcquisitionResult:
        """
        Stop background acquisition and return collected samples.

        If join_timeout_s is None, waits indefinitely.
        If join times out, raises TimeoutError (does NOT return a snapshot),
        because foreground reads are only valid after acquisition is fully stopped.
        """
        if self._acq is None:
            return AcquisitionResult(
                values=self._empty.copy(),
                sample_rate_hz=self.sample_rate_hz,
                t0_wall=time.time(),
            )

        acq = self._acq
        acq.stop_evt.set()

        acq.thread.join(timeout=None if join_timeout_s is None else float(join_timeout_s))

        if acq.thread.is_alive():
            raise TimeoutError(
                "NI6008 acquisition thread did not stop within join_timeout_s. "
                "Acquisition may still be running; call stop_acquiring() again or increase the timeout."
            )

        # Thread fully stopped: safe to finalize.
        with acq.lock:
            n = int(acq.write_idx)
            values = acq.values[:n].copy()
            fs = float(acq.sample_rate_hz)
            t0 = float(acq.t0_wall)
            acq.running = False

        self._acq = None
        return AcquisitionResult(values=values, sample_rate_hz=fs, t0_wall=t0)

    def get_acquiring_snapshot(self) -> Optional[np.ndarray]:
        """
        Get a copy of samples collected so far without stopping acquisition.
        Returns None if not acquiring.
        """
        acq = self._acq
        if acq is None:
            return None
        with acq.lock:
            n = int(acq.write_idx)
            return acq.values[:n].copy()

    # ----------------------------
    # foreground reads (DISABLED while acquiring)
    # ----------------------------

    def get_sample_rate(self) -> float:
        return self.sample_rate_hz

    def read_latest(self, *, timeout_s: float = 0.05) -> float:
        """Read the most recent acquired sample (single value)."""
        self._ensure_streaming()
        self._ensure_not_acquiring()
        assert self._task is not None and self._reader is not None

        self._task.in_stream.relative_to = ReadRelativeTo.MOST_RECENT_SAMPLE
        self._task.in_stream.offset = 0
        return float(self._reader.read_one_sample(timeout=float(timeout_s)))

    def read_block(self, n: int, *, timeout_s: float = 1.0) -> np.ndarray:
        """
        Read up to n samples from FIFO.
        Returned array is a view into an internal reusable buffer and is valid
        only until the next foreground read call.
        """
        self._ensure_streaming()
        self._ensure_not_acquiring()
        assert self._task is not None and self._reader is not None

        n = int(n)
        if n <= 0:
            return self._empty

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        buf = self._ensure_fg_buffer(n)
        view = buf[:n]
        n_read = self._reader.read_many_sample(
            data=view,
            number_of_samples_per_channel=n,
            timeout=float(timeout_s),
        )

        if n_read is None:
            n_read = n
        n_read = int(n_read)

        if n_read <= 0:
            return self._empty

        return view[:n_read] if n_read > 0 else self._empty

    def read_available_block(self, *, timeout_s: float = 0.01) -> np.ndarray:
        """Read all currently available samples from FIFO (may be empty)."""
        self._ensure_streaming()
        self._ensure_not_acquiring()
        assert self._task is not None and self._reader is not None

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
        if avail <= 0:
            return self._empty

        buf = self._ensure_fg_buffer(avail)
        view = buf[:avail]
        n_read = self._reader.read_many_sample(
            data=view,
            number_of_samples_per_channel=avail,
            timeout=float(timeout_s),
        )

        if n_read is None:
            n_read = avail
        n_read = int(n_read)

        if n_read <= 0:
            return self._empty

        return view[:n_read] if n_read > 0 else self._empty

    def flush(self) -> int:
        """Discard all currently buffered samples. Returns number of samples discarded."""
        self._ensure_streaming()
        self._ensure_not_acquiring()
        assert self._task is not None and self._reader is not None

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        total = 0
        while True:
            avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
            if avail <= 0:
                break

            buf = self._ensure_fg_buffer(avail)
            view = buf[:avail]
            try:
                n_read = self._reader.read_many_sample(
                    data=view,
                    number_of_samples_per_channel=avail,
                    timeout=0.0,
                )
            except Exception:
                break

            if n_read is None:
                n_read = avail
            n_read = int(n_read)

            if n_read <= 0:
                break

            total += n_read

        return total


# ----------------------------
# quick manual test
# ----------------------------

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

        # Background acquisition demo (no foreground reads while running)
        ni.start_acquiring(duration_s=2.0, chunk_size=2048)
        time.sleep(0.5)  # do other work

        snap = ni.get_acquiring_snapshot()
        print("snapshot samples:", None if snap is None else snap.size)

        result = ni.stop_acquiring()
        print("acquired samples:", result.values.size)

        ts = result.timestamps()
        if ts.size:
            print("first/last time:", ts[0], ts[-1])