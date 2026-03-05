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


# ----------------------------
# Result type (perf_counter only)
# ----------------------------

@dataclass(frozen=True, slots=True)
class ReadResult:
    """
    Lightweight read result with a perf_counter anchor.

    Timing model (sample-index based):
        time_of(i) = t0_perf + i / sample_rate_hz

    Notes:
      - NI USB-6008 / DAQmx does not provide per-sample hardware timestamps.
      - t0_perf is a best-effort software anchor for acquisition-buffer index 0.
      - ind0 is the acquisition-buffer index corresponding to values[0].
    """
    values: np.ndarray          # shape (n,), float64 (may be a view into internal buffer)
    sample_rate_hz: float       # DAQ sample rate
    t0_perf: float              # perf_counter timestamp for acquisition-buffer sample index 0
    ind0: int                   # acquisition-buffer index corresponding to values[0]

    def time_of(self, i: int) -> float:
        """Return perf_counter time for acquisition-buffer sample index i."""
        return self.t0_perf + (float(i) / float(self.sample_rate_hz))

    def timestamps_perf(self) -> np.ndarray:
        """Return perf_counter timestamps for each returned sample."""
        n = int(self.values.size)
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        fs = float(self.sample_rate_hz)
        return self.time_of(self.ind0) + (np.arange(n, dtype=np.float64) / fs)


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

    # metadata
    sample_rate_hz: float
    t0_perf: float              # best-effort anchor (set on first successful read)

    # diagnostics
    last_error: Optional[str] = None


class NI6008:
    """
    Minimal NI USB-6008 analog input helper.

    Public API:
      - with ni.streaming():  # configure/start DAQmx task, cleanup on exit
      - ni.flush() -> int  # discard buffered FIFO samples (foreground; only when NOT acquiring)
      - ni.start_acquiring(max_samples, chunk_size=..., idle_sleep_s=...)  # background producer
      - ni.get_new_block(last_idx, copy=False) -> ReadResult  # incremental reads
      - ni.get_acquiring_error() -> Optional[str]
      - ni.stop_acquiring() -> ReadResult  # stops producer and returns a copy of acquired samples

    Background acquisition design (fixed buffer):
      - Dedicated daemon thread drains the DAQ FIFO in blocks.
      - Stores values into a fixed preallocated buffer (no growth, no overwrite).
      - Stops when buffer is full or stop is requested.
      - Provides an approximate perf_counter anchor (t0_perf) for sample index 0.
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
        self._empty = np.empty(0, dtype=np.float64)

        self._acq: Optional[_AcqState] = None

    @property
    def physical_channel(self) -> str:
        return f"{self.device}/{self.ai_channel}"

    # ----------------------------
    # lifecycle
    # ----------------------------

    @contextmanager
    def streaming(self):
        """
        Configure and start a continuous DAQmx analog input task.

        Ensures any background acquisition is stopped before closing the task.
        """
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
            # stop background acquisition first
            acq = self._acq
            if acq is not None:
                try:
                    acq.stop_evt.set()
                except Exception:
                    pass
                acq.thread.join(timeout=2.0)
                if acq.thread.is_alive():
                    raise RuntimeError("NI6008 acquisition thread did not stop during streaming() cleanup")
                self._acq = None

            # stop and close task
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
            raise RuntimeError("Foreground operation is disabled while background acquisition is running")

    # ----------------------------
    # background acquisition (fixed buffer)
    # ----------------------------

    def start_acquiring(
        self,
        *,
        max_samples: int,
        chunk_size: int = 2048,
        idle_sleep_s: float = 0.001,
    ) -> None:
        """
        Start background acquisition into a fixed-size buffer.

        Args:
            max_samples:
                Preallocates exactly this many samples. When full, acquisition stops.
            chunk_size:
                Maximum samples read per producer iteration.
            idle_sleep_s:
                Sleep when no samples are available (prevents busy-wait).
        """
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")

        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        idle_sleep_s = float(idle_sleep_s)
        if idle_sleep_s < 0:
            raise ValueError("idle_sleep_s must be >= 0")

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
            t0_perf=0.0,  # set on first successful write
            last_error=None,
        )

        def _producer() -> None:
            tmp = np.empty(chunk_size, dtype=np.float64)
            first_chunk = True

            try:
                if self._task is None or self._reader is None:
                    return

                self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
                self._task.in_stream.offset = 0

                while not stop_evt.is_set():
                    if self._task is None or self._reader is None:
                        break

                    acq = self._acq
                    if acq is None:
                        break

                    # compute room under lock
                    with acq.lock:
                        room = acq.max_samples - acq.write_idx
                        if room <= 0:
                            stop_evt.set()
                            break

                    avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
                    if avail <= 0:
                        if idle_sleep_s > 0:
                            time.sleep(idle_sleep_s)
                        continue

                    want = min(room, chunk_size, avail)
                    if want <= 0:
                        continue

                    try:
                        n_read = self._reader.read_many_sample(
                            data=tmp[:want],
                            number_of_samples_per_channel=want,
                            timeout=0.02,
                        )
                    except Exception as e:
                        with lock:
                            acq2 = self._acq
                            if acq2 is not None:
                                acq2.last_error = repr(e)
                        stop_evt.set()
                        break

                    if n_read is None:
                        n_read = want
                    n_read = int(n_read)
                    if n_read <= 0:
                        continue

                    t_after = time.perf_counter()

                    with lock:
                        acq2 = self._acq
                        if acq2 is None:
                            break

                        room2 = acq2.max_samples - acq2.write_idx
                        take = n_read if n_read <= room2 else room2
                        if take <= 0:
                            stop_evt.set()
                            break

                        acq2.values[acq2.write_idx: acq2.write_idx + take] = tmp[:take]

                        if first_chunk:
                            # Best-effort anchor: time_of(last_written_index) ~= t_after
                            last_i = acq2.write_idx + take - 1
                            acq2.t0_perf = t_after - (last_i / acq2.sample_rate_hz)
                            first_chunk = False

                        acq2.write_idx += take

                        if take < n_read:
                            stop_evt.set()
                            break

            finally:
                with lock:
                    acq2 = self._acq
                    if acq2 is not None:
                        acq2.running = False

        th = threading.Thread(target=_producer, name="NI6008Acq", daemon=True)
        self._acq.thread = th
        th.start()

    def get_new_block(self, last_idx: int, *, copy: bool = False) -> ReadResult:
        """
        Return the new samples since last_idx while acquisition is running.

        Cursor usage:
            blk = ni.get_new_block(last)
            last = blk.ind0 + blk.values.size

        Args:
            last_idx:
                Acquisition-buffer index to start from.
            copy:
                If True, return an owned copy; otherwise return a view into the internal buffer.

        Returns:
            ReadResult with:
              - values: samples in [last_idx, write_idx)
              - ind0: clamped start index
              - t0_perf: acquisition-buffer index 0 anchor
        """
        acq = self._acq
        if acq is None:
            now = time.perf_counter()
            return ReadResult(self._empty, float(self.sample_rate_hz), now, int(last_idx))

        start = int(last_idx)
        if start < 0:
            start = 0

        with acq.lock:
            end = int(acq.write_idx)
            t0 = float(acq.t0_perf)
            fs = float(acq.sample_rate_hz)

            if start >= end:
                return ReadResult(self._empty, fs, t0, start)

            view = acq.values[start:end]
            out = view.copy() if copy else view
            return ReadResult(out, fs, t0, start)

    def get_acquiring_error(self) -> Optional[str]:
        """If the producer hit an exception, returns repr(exception)."""
        acq = self._acq
        if acq is None:
            return None
        with acq.lock:
            return acq.last_error

    def stop_acquiring(self, *, join_timeout_s: float | None = 2.0) -> ReadResult:
        """
        Stop background acquisition and return collected samples (copied).

        If join_timeout_s is None, waits indefinitely.
        If join times out, raises TimeoutError.
        """
        acq = self._acq
        if acq is None:
            now = time.perf_counter()
            return ReadResult(self._empty.copy(), float(self.sample_rate_hz), now, 0)

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
        return ReadResult(values=values, sample_rate_hz=fs, t0_perf=t0, ind0=0)

    # ----------------------------
    # optional foreground helper
    # ----------------------------

    def flush(self, *, chunk_size: int = 4096) -> int:
        """
        Discard all currently buffered FIFO samples.

        Foreground only: raises if background acquisition is running.
        Returns the number of samples discarded.
        """
        self._ensure_streaming()
        self._ensure_not_acquiring()
        assert self._task is not None and self._reader is not None

        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        buf = np.empty(chunk_size, dtype=np.float64)
        total = 0

        while True:
            avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
            if avail <= 0:
                break

            want = avail if avail < chunk_size else chunk_size
            view = buf[:want]

            try:
                n_read = self._reader.read_many_sample(
                    data=view,
                    number_of_samples_per_channel=want,
                    timeout=0.0,
                )
            except Exception:
                break

            if n_read is None:
                n_read = want

            n_read = int(n_read)
            if n_read <= 0:
                break

            total += n_read

        return total


    def read_latest(self, *, timeout_s: float = 0.05) -> float:
        """
        Return the most recent sample available from the device.

        This moves the DAQmx read pointer to the most recent sample and
        reads a single value, effectively discarding any older samples
        currently buffered in the device FIFO.

        This is useful when only the newest value matters and historical
        samples should be ignored.

        Args:
            timeout_s:
                Maximum time to wait for a sample.

        Returns:
            Latest voltage sample as a float.
        """
        self._ensure_streaming()
        self._ensure_not_acquiring()

        assert self._task is not None and self._reader is not None

        self._task.in_stream.relative_to = ReadRelativeTo.MOST_RECENT_SAMPLE
        self._task.in_stream.offset = 0

        return float(self._reader.read_one_sample(timeout=float(timeout_s)))


    # ---------- FIFO block ----------

    def read_block(self, n: int, *, timeout_s: float = 1.0) -> list[float]:
        """
        Read exactly `n` sequential samples from the device FIFO.

        The read starts from the current DAQmx read position and advances
        the read pointer by `n` samples.

        This call blocks until all requested samples are available or the
        timeout expires.

        Args:
            n:
                Number of samples to read.
            timeout_s:
                Maximum time to wait for the full block.

        Returns:
            List of `n` float samples.
        """
        self._ensure_streaming()
        self._ensure_not_acquiring()

        assert self._task is not None

        n = int(n)
        if n <= 0:
            return []

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        data = self._task.read(
            number_of_samples_per_channel=n,
            timeout=float(timeout_s),
        )

        if isinstance(data, float):
            return [float(data)]

        return [float(x) for x in data]


    def read_available_block(self) -> list[float]:
        """
        Read all currently available samples from the device FIFO.

        This call is non-blocking. If no samples are available, an empty
        list is returned.

        The read pointer advances by the number of samples returned.

        Returns:
            List of float samples currently available in the FIFO.
        """
        self._ensure_streaming()
        self._ensure_not_acquiring()

        assert self._task is not None

        self._task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
        self._task.in_stream.offset = 0

        avail = int(getattr(self._task.in_stream, "avail_samp_per_chan", 0))
        if avail <= 0:
            return []

        data = self._task.read(
            number_of_samples_per_channel=avail,
            timeout=0.01,
        )

        if isinstance(data, float):
            return [float(data)]

        return [float(x) for x in data]



# ----------------------------
# quick manual test (only uses methods defined here)
# ----------------------------

if __name__ == "__main__":
    ni = NI6008(device="Dev1", ai_channel="ai0", sample_rate_hz=1000)

    with ni.streaming():
        time.sleep(0.2)

        # ---- flush ----
        n_flushed = ni.flush()
        print("flushed:", n_flushed)

        # ---- read_latest ----
        t1 = time.perf_counter()
        x_latest = ni.read_latest(timeout_s=0.05)
        t2 = time.perf_counter()
        print(f"read_latest: {x_latest:.6f}  (dt={(t2 - t1):.6f} s)")

        # ---- read_block ----
        # give the FIFO a moment to accumulate
        time.sleep(0.05)
        t1 = time.perf_counter()
        blk10 = ni.read_block(10, timeout_s=1.0)
        t2 = time.perf_counter()
        print(f"read_block(10): n={len(blk10)} dt={(t2 - t1):.6f} s std={float(np.std(blk10)) if blk10 else float('nan')}")

        # ---- read_available_block ----
        # let some samples arrive, then read whatever is there (non-blocking)
        time.sleep(0.05)
        t1 = time.perf_counter()
        avail_blk = ni.read_available_block()
        t2 = time.perf_counter()
        print(f"read_available_block: n={len(avail_blk)} dt={(t2 - t1):.6f} s")

        # optional: drain again so acquisition starts "clean"
        print("flushed:", ni.flush())

        # ---- background acquisition ----
        max_samples = int(ni.sample_rate_hz * 2.0)  # 2 seconds
        ni.start_acquiring(max_samples=max_samples, chunk_size=2048, idle_sleep_s=0.001)

        last = 0
        for _ in range(5):
            time.sleep(0.1)
            rr = ni.get_new_block(last, copy=False)

            n_new = int(rr.values.size)
            last = rr.ind0 + n_new

            if n_new:
                t_first = rr.time_of(rr.ind0)
                t_last = rr.time_of(rr.ind0 + n_new - 1)
                std = float(np.std(rr.values))
            else:
                t_first = None
                t_last = None
                std = float("nan")

            print("get_new_block:",
                  "new:", n_new,
                  "cursor:", last,
                  "t_first/t_last:", t_first, t_last,
                  "std:", std)

        # capture error BEFORE stop clears acquisition state
        err = ni.get_acquiring_error()

        result = ni.stop_acquiring()
        print("stop_acquiring: acquired:", int(result.values.size))

        if err is not None:
            print("acq error:", err)

        if result.values.size:
            print("first/last perf:",
                  result.time_of(0),
                  result.time_of(int(result.values.size) - 1))
            print("std:", float(np.std(result.values)))