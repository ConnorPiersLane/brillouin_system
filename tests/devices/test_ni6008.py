"""
Manual integration test for NI6008 helper.

What it tests:
  - streaming() lifecycle
  - get_sample_rate()
  - flush()
  - read_latest()
  - read_block(n)
  - read_available_block()
  - start_acquiring()
  - get_acquiring_snapshot()
  - stop_acquiring()
  - timestamps()
  - foreground-read protection while acquiring (should raise RuntimeError)

Run:
  python ni6008_test.py
"""

from __future__ import annotations

import time
import numpy as np

# Import your class (adjust path as needed)
from brillouin_system.devices.ni.ni6008 import NI6008


def expect_raises(fn, exc_type: type[BaseException], label: str) -> None:
    try:
        fn()
    except exc_type as e:
        print(f"[OK] {label}: raised {exc_type.__name__}: {e}")
        return
    except Exception as e:
        raise AssertionError(f"[FAIL] {label}: raised wrong exception {type(e).__name__}: {e}") from e
    raise AssertionError(f"[FAIL] {label}: did NOT raise {exc_type.__name__}")


def main() -> None:
    ni = NI6008(device="Dev1", ai_channel="ai0", sample_rate_hz=1000)

    with ni.streaming():
        print("\n=== streaming() entered ===")
        fs = ni.get_sample_rate()
        print("sample_rate_hz:", fs)

        # Let buffer fill a bit
        time.sleep(0.2)

        # flush()
        n_flushed = ni.flush()
        print("flush() discarded:", n_flushed)

        # read_latest()
        t1 = time.monotonic()
        x_latest = ni.read_latest(timeout_s=0.2)
        t2 = time.monotonic()
        print(f"read_latest(): {x_latest:.6f}  (dt={t2 - t1:.6f}s)")

        # read_block(1) and read_block(100)
        x1 = ni.read_block(1, timeout_s=0.2)
        print("read_block(1): size=", x1.size, " value=", (float(x1[0]) if x1.size else None))

        x100 = ni.read_block(100, timeout_s=0.5)
        print("read_block(100): size=", x100.size, " std=", float(np.std(x100)) if x100.size else None)

        # read_available_block()
        xav = ni.read_available_block(timeout_s=0.05)
        print("read_available_block(): size=", xav.size)

        # flush again
        n_flushed2 = ni.flush()
        print("flush() discarded:", n_flushed2)

        # ----------------------------
        # Background acquisition tests
        # ----------------------------
        print("\n=== background acquisition ===")
        ni.start_acquiring(duration_s=2.0, chunk_size=2048, poll_timeout_s=0.1)

        # Foreground reads should be disabled while acquiring
        expect_raises(lambda: ni.read_latest(), RuntimeError, "read_latest() while acquiring")
        expect_raises(lambda: ni.read_block(10), RuntimeError, "read_block() while acquiring")
        expect_raises(lambda: ni.read_available_block(), RuntimeError, "read_available_block() while acquiring")
        expect_raises(lambda: ni.flush(), RuntimeError, "flush() while acquiring")

        # Let it run
        time.sleep(0.5)

        snap = ni.get_acquiring_snapshot()
        print("get_acquiring_snapshot():", None if snap is None else snap.size, "samples")
        if snap is not None and snap.size:
            print("  snapshot std:", float(np.std(snap)))

        # Stop and validate result
        result = ni.stop_acquiring(join_timeout_s=5.0)
        print("stop_acquiring():", result.values.size, "samples")

        # timestamps()
        ts = result.timestamps()
        if ts.size:
            print("timestamps(): first=", ts[0], " last=", ts[-1])
            dur = ts[-1] - ts[0]
            print("  reconstructed duration (s):", dur)
            print("  expected duration ~= (n-1)/fs:", (result.values.size - 1) / result.sample_rate_hz if result.values.size else 0.0)

        # After stop, foreground reads should work again
        x_after = ni.read_block(10, timeout_s=0.5)
        print("read_block(10) after stop: size=", x_after.size)

        print("\n=== streaming() exiting ===")

    print("Done.")


if __name__ == "__main__":
    main()