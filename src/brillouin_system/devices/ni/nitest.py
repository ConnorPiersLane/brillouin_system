"""
NI USB-6008 quick-connect example (NI-DAQmx + nidaqmx)

What it does:
1) Lists detected NI-DAQmx devices
2) Reads one analog input sample from ai0
3) Writes one analog output value to ao0 (USB-6008 has 2 AO channels)

Notes:
- Device name is often "Dev1" (but can differ). This script auto-picks the first device.
- AI range for USB-6008 is typically ±10 V (depends on config); adjust as needed.
"""

import sys
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.system import System


def pick_device_name() -> str:
    system = System.local()
    devices = list(system.devices)

    if not devices:
        raise RuntimeError(
            "No NI-DAQmx devices found. "
            "Check that NI-DAQmx is installed and the USB-6008 is plugged in."
        )

    print("Detected NI-DAQmx devices:")
    for d in devices:
        # d.name is like "Dev1"; d.product_type might say "USB-6008"
        print(f"  - {d.name}: {d.product_type} (S/N: {getattr(d, 'serial_num', 'unknown')})")

    # Prefer the USB-6008 if multiple devices exist
    for d in devices:
        if "6008" in (d.product_type or ""):
            print(f"\nUsing device: {d.name} ({d.product_type})")
            return d.name

    print(f"\nUsing first device: {devices[0].name} ({devices[0].product_type})")
    return devices[0].name


def read_ai_single_sample(dev: str, channel: str = "ai0") -> float:
    physical_channel = f"{dev}/{channel}"
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            physical_channel,
            min_val=-10.0,
            max_val=10.0,
        )
        # Single read (on-demand)
        value = task.read()
    return float(value)


def write_ao_single_value(dev: str, channel: str = "ao0", volts: float = 1.0) -> None:
    physical_channel = f"{dev}/{channel}"
    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan(
            physical_channel,
            min_val=0.0,
            max_val=5.0,  # USB-6008 AO is typically 0–5 V
        )
        task.write(volts, auto_start=True)


def main():
    try:
        dev = pick_device_name()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # --- Analog Input: read ai0 ---
    try:
        v = read_ai_single_sample(dev, "ai0")
        print(f"\nAI read from {dev}/ai0: {v:.4f} V")
    except Exception as e:
        print(f"\nAI read failed: {e}")

    # --- Analog Output: write ao0 ---
    try:
        out_v = 1.0
        write_ao_single_value(dev, "ao0", out_v)
        print(f"AO wrote to {dev}/ao0: {out_v:.3f} V")
        time.sleep(0.2)
        # Optional: set back to 0 V
        write_ao_single_value(dev, "ao0", 0.0)
        print(f"AO reset {dev}/ao0 to 0.000 V")
    except Exception as e:
        print(f"\nAO write failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
