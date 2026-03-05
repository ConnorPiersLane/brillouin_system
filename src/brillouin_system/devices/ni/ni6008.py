from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, ReadRelativeTo, TerminalConfiguration
from nidaqmx.stream_readers import AnalogSingleChannelReader

from brillouin_system.devices.ni.ni_base import NIBase


# ----------------------------
# Result type (perf_counter only)
# ----------------------------

@dataclass(frozen=True, slots=True)
class ReadResult:
    """
    Lightweight read result with a perf_counter anchor.

    Timing model:
      t_perf[i] = t0_perf + i / sample_rate_hz

    Note:
      The NI USB-6008 / DAQmx does not provide hardware timestamps per sample.
      t0_perf is a best-effort software anchor.
    """
    values: np.ndarray          # shape (n,), float64 (may be view into internal buffer)
    sample_rate_hz: float       # DAQ sample rate
    t0_perf: float              # perf_counter timestamp for sample index 0 of `values`

    def timestamps_perf(self) -> np.ndarray:
        n = int(self.values.size)
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        fs = float(self.sample_rate_hz)
        return self.t0_perf + (np.arange(n, dtype=np.float64) / fs)

    def time_of(self, i: int) -> float:
        return self.t0_perf + (float(i) / float(self.sample_rate_hz))


@dataclass
class _AcqState:
    stop_evt: threading.Event
    thread: threading.Thread
    lock: threading.Lock
    running: bool

    # fixed storage
    values: np.ndarray          # fixed-size preallocated buffer
    write_idx: int              # next write position
    max_samples: int            # capacity (hard cap)

    # metadata (perf only)
    sample_rate_hz: float
    t0_perf: float              # set on first successful read (best-effort anchor)
    t_last_perf: float          # updated on each read loop

    # diagnostics
    last_error: Optional[str] = None


# ----------------------------
# NI6008 device
# ----------------------------

class NI6008(NIBase):
    """
    Minimal NI USB-6008 analog input helper (optimized).

    Foreground reads (read_latest/read_block/read_available_block/flush) are allowed
    ONLY when background acquisition is NOT running.

    Background acquisition design (simplified, fixed buffer):
      - Dedicated daemon thread continuously drains the DAQ FIFO in blocks.
      - Stores ONLY values into a fixed preallocated buffer (no growth, no overwrite).
      - Stops when buffer is full.
      - Uses perf_counter only for timing anchor (no time.time()).
      - Provides incremental reads via get_new_samples(last_idx).
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
    # background acquisition (fixed buffer)
    # ----------------------------

    def start_acquiring(
        self,
        *,
        max_samples: int,
        chunk_size: int = 2048,
    ) -> None:
        """
        Start continuous background acquisition into a fixed-size buffer.

        Args:
            max_samples:
                Required. Preallocates exactly this many samples. When full, acquisition stops.
            chunk_size:
                Max samples to read per producer loop iteration.
            idle_sleep_s:
                Sleep duration when no samples are available (reduces CPU busy-wait).
        """
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")

        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")


        if self._acq is not None and self._acq.running:
            raise RuntimeError("Acquisition already running")

        self._ensure_streaming()
        assert self._task is not None and self._reader is not None

        fs = float(self.sample_rate_hz)
        values = np.empty(max_samples, dtype=np.float64)

        stop_evt = threading.Event()
        lock = threading.Lock()

        self._acq = _AcqState(
            stop_evt=stop_evt,
            thread=threading.Thread(),  # placeholder
            lock=lock,
            running=True,
            values=values,
            write_idx=0,
            max_samples=max_samples,
            sample_rate_hz=fs,
            t0_perf=0.0,       # will be refined on first successful read
            t_last_perf=0.0,
            last_error=None,
        )

        def _producer() -> None:
            tmp = np.empty(chunk_size, dtype=np.float64)
            first_chunk = True

            try:
                if self._task is None or self._reader is None:
                    return

                # Read from FIFO current read position.
                self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
                self._task.in_stream.offset = 0

                while not stop_evt.is_set():
                    if self._task is None or self._reader is None:
                        break

                    # Determine how many we can still store
                    with self._acq.lock:
                        acq = self._acq
                        if acq is None:
                            break
                        room = acq.max_samples - acq.write_idx
                        if room <= 0:
                            stop_evt.set()
                            break

                    avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
                    if avail <= 0:
                        continue  # or time.sleep(0.001)

                    want = min(room, chunk_size, avail)

                    # Block in DAQmx briefly instead of polling avail
                    try:
                        n_read = self._reader.read_many_sample(
                            data=tmp[:want],
                            number_of_samples_per_channel=want,
                            timeout=0.02,  # 20 ms; tune 5–50 ms
                        )
                    except Exception as e:
                        with lock:
                            acq = self._acq
                            if acq is not None:
                                acq.last_error = repr(e)
                        stop_evt.set()
                        break

                    if n_read is None:
                        n_read = want
                    n_read = int(n_read)
                    if n_read <= 0:
                        continue

                    t_last = time.perf_counter()

                    with lock:
                        acq = self._acq
                        if acq is None:
                            break

                        room = acq.max_samples - acq.write_idx
                        take = n_read if n_read <= room else room

                        acq.values[acq.write_idx: acq.write_idx + take] = tmp[:take]

                        if first_chunk and take > 0:
                            acq.t0_perf = t_last - (take - 1) / acq.sample_rate_hz
                            first_chunk = False

                        acq.write_idx += take
                        acq.t_last_perf = t_last

                        if take < n_read:
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

    def stop_acquiring(self, *, join_timeout_s: float | None = 2.0) -> ReadResult:
        """
        Stop background acquisition and return collected samples.

        If join_timeout_s is None, waits indefinitely.
        If join times out, raises TimeoutError.
        """
        acq = self._acq
        if acq is None:
            now = time.perf_counter()
            return ReadResult(values=self._empty.copy(), sample_rate_hz=float(self.sample_rate_hz), t0_perf=now)

        acq.stop_evt.set()

        acq.thread.join(timeout=None if join_timeout_s is None else float(join_timeout_s))
        if acq.thread.is_alive():
            raise TimeoutError(
                "NI6008 acquisition thread did not stop within join_timeout_s. "
                "Acquisition may still be running; call stop_acquiring() again or increase the timeout."
            )

        with acq.lock:
            n = int(acq.write_idx)
            values = acq.values[:n].copy()
            fs = float(acq.sample_rate_hz)
            t0 = float(acq.t0_perf)

        self._acq = None
        return ReadResult(values=values, sample_rate_hz=fs, t0_perf=t0)

    def get_new_samples(self, last_idx: int, *, copy: bool = False) -> Tuple[np.ndarray, int]:
        """
        Return samples in [last_idx:write_idx) and the updated index.

        Because the acquisition buffer is fixed-size and never reallocated and never overwritten,
        returning a view (copy=False) is safe as long as the caller does not expect the view
        to remain valid after acquisition ends/cleans up.

        Args:
            last_idx:
                Your previous cursor/index (typically 0 initially).
            copy:
                If True, returns a copy. If False, returns a view into the fixed buffer.
        """
        acq = self._acq
        if acq is None:
            return self._empty, int(last_idx)

        start = int(last_idx)
        if start < 0:
            start = 0


        with acq.lock:
            end = int(acq.write_idx)
            if start >= end:
                return self._empty, end
            view = acq.values[start:end]
            out = view.copy() if copy else view
            return out, end

    def get_new_block(self, last_idx: int, *, copy: bool = False) -> Tuple[ReadResult, int]:
        """
        Convenience wrapper: returns a ReadResult for the new samples since last_idx,
        where t0_perf corresponds to the first sample in the returned block.

        Useful for syncing to other perf_counter-based logs (e.g., Zaber polling).
        """
        acq = self._acq
        if acq is None:
            now = time.perf_counter()
            return ReadResult(self._empty, float(self.sample_rate_hz), now), int(last_idx)

        xs, new_last = self.get_new_samples(last_idx, copy=copy)
        if xs.size == 0:
            with acq.lock:
                fs = float(acq.sample_rate_hz)
                # Anchor to the time of where the next sample would be, for consistency.
                t0 = float(acq.t0_perf) + (float(new_last) / fs)
            return ReadResult(xs, fs, t0), new_last

        with acq.lock:
            fs = float(acq.sample_rate_hz)
            t0_acq = float(acq.t0_perf)

        start = int(new_last - xs.size)
        t0_block = t0_acq + (start / fs)
        return ReadResult(xs, fs, t0_block), new_last

    def get_acquiring_error(self) -> Optional[str]:
        """If the producer hit an exception, returns repr(exception)."""
        acq = self._acq
        if acq is None:
            return None
        with acq.lock:
            return acq.last_error

    # ----------------------------
    # foreground reads (DISABLED while acquiring)
    # ----------------------------

    def get_sample_rate(self) -> float:
        return float(self.sample_rate_hz)

    def read_latest(self, *, timeout_s: float = 0.05) -> ReadResult:
        """
        Read the most recent acquired sample.

        Note: DAQmx does not provide true ADC timestamps for this sample.
        We provide a best-effort perf_counter anchor using the convention:
          t0_perf ~ time of the returned sample (host-side).
        """
        self._ensure_streaming()
        self._ensure_not_acquiring()
        assert self._task is not None and self._reader is not None

        self._task.in_stream.relative_to = ReadRelativeTo.MOST_RECENT_SAMPLE
        self._task.in_stream.offset = 0

        buf = self._ensure_fg_buffer(1)
        view = buf[:1]
        n_read = self._reader.read_many_sample(
            data=view,
            number_of_samples_per_channel=1,
            timeout=float(timeout_s),
        )
        if n_read is None:
            n_read = 1
        n_read = int(n_read)

        if n_read <= 0:
            now = time.perf_counter()
            return ReadResult(values=self._empty, sample_rate_hz=float(self.sample_rate_hz), t0_perf=now)

        t_last = time.perf_counter()
        return ReadResult(values=view[:1], sample_rate_hz=float(self.sample_rate_hz), t0_perf=t_last)

    def read_block(self, n: int, *, timeout_s: float = 1.0) -> ReadResult:
        """
        Read up to n samples from FIFO.

        Returned values are a view into an internal reusable buffer and are valid
        only until the next foreground read call.

        Timing anchor uses convention:
          t_last = perf_counter() right after read call returns
          treat t_last as time of last sample in returned block
          t0 = t_last - (n_read - 1) / fs
        """
        self._ensure_streaming()
        self._ensure_not_acquiring()
        assert self._task is not None and self._reader is not None

        n = int(n)
        if n <= 0:
            now = time.perf_counter()
            return ReadResult(values=self._empty, sample_rate_hz=float(self.sample_rate_hz), t0_perf=now)

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
            now = time.perf_counter()
            return ReadResult(values=self._empty, sample_rate_hz=float(self.sample_rate_hz), t0_perf=now)

        t_last = time.perf_counter()
        fs = float(self.sample_rate_hz)
        t0 = t_last - (n_read - 1) / fs

        return ReadResult(values=view[:n_read], sample_rate_hz=fs, t0_perf=t0)

    def read_available_block(self, *, timeout_s: float = 0.01) -> ReadResult:
        """
        Read all currently available samples from FIFO (may be empty).

        Returned values are a view into an internal reusable buffer and are valid
        only until the next foreground read call.
        """
        self._ensure_streaming()
        self._ensure_not_acquiring()
        assert self._task is not None and self._reader is not None

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
        if avail <= 0:
            now = time.perf_counter()
            return ReadResult(values=self._empty, sample_rate_hz=float(self.sample_rate_hz), t0_perf=now)

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
            now = time.perf_counter()
            return ReadResult(values=self._empty, sample_rate_hz=float(self.sample_rate_hz), t0_perf=now)

        t_last = time.perf_counter()
        fs = float(self.sample_rate_hz)
        t0 = t_last - (n_read - 1) / fs

        return ReadResult(values=view[:n_read], sample_rate_hz=fs, t0_perf=t0)

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

        rr = ni.read_block(100)
        print("std(100):", float(np.std(rr.values)))
        print("block t0_perf:", rr.t0_perf, "t_last:", rr.time_of(rr.values.size - 1) if rr.values.size else None)

        print("flushed:", ni.flush())

        t1 = time.perf_counter()
        _ = ni.read_latest()
        t2 = time.perf_counter()
        print(f"read_latest time: {t2 - t1:.6f} s")

        t1 = time.perf_counter()
        _ = ni.read_block(1)
        t2 = time.perf_counter()
        print(f"read_block(1) time: {t2 - t1:.6f} s")

        # Background acquisition demo (no foreground reads while running)
        max_samples = int(ni.sample_rate_hz * 2.0)  # 2 seconds
        ni.start_acquiring(max_samples=max_samples, chunk_size=2048)

        last = 0
        for _ in range(5):
            time.sleep(0.1)
            xs, last = ni.get_new_samples(last, copy=False)
            print("new samples:", xs.size, "cursor:", last)

        result = ni.stop_acquiring()
        print("acquired samples:", result.values.size)
        if result.values.size:
            print("first/last perf:", result.t0_perf, result.time_of(result.values.size - 1))