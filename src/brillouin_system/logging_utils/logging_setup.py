# logging_setup.py
"""
Robust, crash-resilient logging with a dedicated writer process.
- Main/GUI process logs via QueueHandler to a non-daemon writer process.
- Writer owns a RotatingFileHandler with fsync on every emit (durability).
- Safe on Windows (spawn) with proper __main__ guard requirements.
- Optional Qt handler support for GUI log view.
"""

from __future__ import annotations
import atexit
import logging
import os
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue, get_start_method, current_process
from logging.handlers import RotatingFileHandler, QueueHandler

# ------------------------- Paths & constants -------------------------

LOG_DIR = Path.home() / "BrillouinLogs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"brillouin_{timestamp}.log"

# Optional crash mirror (for last-resort traceback writes)
CRASH_PATH = LOG_DIR / f"crash_{timestamp}.log"

# Main logger name used across the app
LOGGER_NAME = "brillouin"

logging_fmt_gui = logging.Formatter("[%(levelname)s] %(message)s")
logging_fmt_file = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s %(message)s")

# ------------------------- Module-level state -------------------------

_queue: Queue | None = None
_writer_proc: Process | None = None


# ------------------------- Writer process target -------------------------

def _writer_main(queue: Queue, log_path: str):
    """
    Runs in a separate process. Receives LogRecord objects from the Queue and
    writes them to disk with fsync for durability.
    """
    import time

    class FsyncRotatingFileHandler(RotatingFileHandler):
        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            try:
                self.flush()
                if self.stream and hasattr(self.stream, "fileno"):
                    os.fsync(self.stream.fileno())
            except Exception:
                # Never raise from logging
                pass

    # Configure minimal logging in the writer
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    fh = FsyncRotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(logging_fmt_file)
    log.addHandler(fh)

    try:
        while True:
            rec = queue.get()  # blocking
            if rec == "__STOP__":
                break
            # Forward the record to the file handler(s)
            log.handle(rec)
    except Exception:
        # Last-resort crash mirror
        try:
            with open(CRASH_PATH, "a", buffering=1, encoding="utf-8") as f:
                f.write(f"Writer crashed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception:
            pass
    finally:
        for h in list(log.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass


# ------------------------- Public API -------------------------

def start_logging() -> None:
    """
    Start the dedicated writer process and install a QueueHandler in the current process.
    MUST be called once from the main (GUI) process, inside:
        if __name__ == "__main__":
            start_logging()
    """
    global _queue, _writer_proc

    # Avoid double-start
    if _queue is not None:
        return

    # Only the main process is allowed to spawn the writer
    if current_process().name != "MainProcess":
        return

    # Ensure Windows-safe start method
    try:
        if get_start_method(allow_none=True) != "spawn":
            import multiprocessing as mp
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set; that's fine
        pass

    # Use multiprocessing.Queue (supports put_nowait)
    _queue = Queue()

    _writer_proc = Process(
        target=_writer_main,
        args=(_queue, str(LOG_PATH)),
        name="LogWriter",
    )
    _writer_proc.daemon = False  # critical: keep it alive independently
    _writer_proc.start()

    _install_queue_handler()

    # Try to shutdown cleanly on normal interpreter exit
    atexit.register(shutdown_logging)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return the root app logger ('brillouin') or a child under it,
    so all children inherit the QueueHandler attached to 'brillouin'.
    """
    if _queue is not None:
        _install_queue_handler()

    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)

    # Child under 'brillouin' hierarchy:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")



def shutdown_logging(timeout: float = 2.0) -> None:
    """
    Ask the writer to stop, flush, and close. Safe to call multiple times.
    """
    global _queue, _writer_proc

    # Remove QueueHandler from this process to stop enqueuing
    _remove_queue_handlers()

    if _queue is not None:
        try:
            _queue.put("__STOP__")
        except Exception:
            pass

    if _writer_proc is not None:
        try:
            _writer_proc.join(timeout)
            if _writer_proc.is_alive():
                _writer_proc.terminate()
        except Exception:
            try:
                _writer_proc.terminate()
            except Exception:
                pass

    _writer_proc = None
    _queue = None

    # Close any remaining handlers in this process (e.g., Qt handler)
    logging.shutdown()


def install_crash_hooks() -> None:
    """
    Mirrors uncaught exceptions to the main logger and also to CRASH_PATH for redundancy.
    Call this once in your entry script after start_logging().
    """
    import sys, threading, traceback, faulthandler

    try:
        # Enables low-level tracebacks on hard crashes
        faulthandler.enable(open(CRASH_PATH, "a", buffering=1, encoding="utf-8"))
    except Exception:
        pass

    def _excepthook(exc_type, exc, tb):
        log = get_logger()
        log.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc, tb))
        try:
            with open(CRASH_PATH, "a", buffering=1, encoding="utf-8") as f:
                traceback.print_exception(exc_type, exc, tb, file=f)
        except Exception:
            pass

    def _thread_excepthook(args):
        _excepthook(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook


# ------------------------- Optional: Qt GUI log handler -------------------------

class _QtTextEditHandler(logging.Handler):
    """
    Lightweight handler that emits formatted log lines through a Qt signal bridge.
    Create the bridge in your Qt code and pass it here to mirror logs into a QTextEdit.
    """
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # The bridge is expected to be a QObject with a `message` pyqtSignal(str)
            self.bridge.message.emit(msg)
        except Exception:
            self.handleError(record)


def install_qt_gui_handler(bridge, fmt: str = "%(asctime)s  %(message)s") -> None:
    """
    Add a Qt text handler so logs appear in your GUI log panel.
    `bridge` must expose a pyqtSignal[str] named `message`.
    Safe to call multiple times; only installs once.
    """
    lg = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(h, _QtTextEditHandler) for h in lg.handlers):
        h = _QtTextEditHandler(bridge)
        h.setFormatter(logging.Formatter(fmt))
        lg.addHandler(h)


# ------------------------- Internal helpers -------------------------

def _install_queue_handler() -> None:
    if _queue is None:
        return
    lg = logging.getLogger(LOGGER_NAME)
    lg.setLevel(logging.INFO)
    if not any(isinstance(h, QueueHandler) for h in lg.handlers):
        qh = QueueHandler(_queue)  # no formatter here ✅
        lg.addHandler(qh)



def _remove_queue_handlers() -> None:
    lg = logging.getLogger(LOGGER_NAME)
    for h in list(lg.handlers):
        if isinstance(h, QueueHandler):
            try:
                lg.removeHandler(h)
                h.flush()
                h.close()
            except Exception:
                pass
