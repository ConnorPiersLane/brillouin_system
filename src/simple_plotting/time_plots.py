from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

path = Path("data_scan.txt")   # adjust if needed

text = path.read_text(encoding="utf-8")

def parse_scalar(name: str, text: str) -> float:
    m = re.search(rf"\b{name}\s*=\s*([-+0-9.eE]+)", text)
    if not m:
        raise ValueError(f"Could not find scalar {name}")
    return float(m.group(1))

def parse_array(name: str, text: str) -> np.ndarray:
    # grab everything inside the outer [...]
    m = re.search(rf"\b{name}\s*=\s*\[(.*?)\]", text, re.S)
    if not m:
        raise ValueError(f"Could not find array {name}")
    body = m.group(1)

    # extract all floats regardless of commas / whitespace formatting
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", body)
    return np.asarray([float(x) for x in nums], dtype=float)

# --- load everything ---
t0 = parse_scalar("t0", text)
f_event_ts = parse_scalar("f_event_ts", text)
b_event_ts = parse_scalar("b_event_ts", text)

f_daq_ts  = parse_array("f_daq_ts", text)
f_daq_v   = parse_array("f_daq_v", text)
f_event_z = parse_scalar("f_event_z", text)
f_zaber_ts  = parse_array("f_zaber_ts", text)
f_zaber_pos = parse_array("f_zaber_pos", text)

b_daq_ts  = parse_array("b_daq_ts", text)
b_daq_v   = parse_array("b_daq_v", text)
b_event_z = parse_scalar("b_event_z", text)
b_zaber_ts  = parse_array("b_zaber_ts", text)
b_zaber_pos = parse_array("b_zaber_pos", text)



print("Loaded:")
print("t0:", t0)
print("f_daq:", len(f_daq_ts), len(f_daq_v))
print("f_zaber:", len(f_zaber_ts), len(f_zaber_pos))
print("b_daq:", len(b_daq_ts), len(b_daq_v))
print("b_zaber:", len(b_zaber_ts), len(b_zaber_pos))

# --- convert to arrays ---
f_daq_ts = np.asarray(f_daq_ts, dtype=float)
f_daq_v = np.asarray(f_daq_v, dtype=float)

b_daq_ts = np.asarray(b_daq_ts, dtype=float)
b_daq_v = np.asarray(b_daq_v, dtype=float)

f_zaber_ts = np.asarray(f_zaber_ts, dtype=float)
f_zaber_pos = np.asarray(f_zaber_pos, dtype=float)
f_event_ts = np.asarray(f_event_ts, dtype=float)
b_zaber_ts = np.asarray(b_zaber_ts, dtype=float)
b_zaber_pos = np.asarray(b_zaber_pos, dtype=float)
b_event_ts = np.asarray(b_event_ts, dtype=float)
# --- shift time so t0 is approximately 0 ---
f_daq_t = f_daq_ts - t0
b_daq_t = b_daq_ts - t0
f_zaber_t = f_zaber_ts - t0
b_zaber_t = b_zaber_ts - t0

f_event_t = f_event_ts - t0
b_event_t = b_event_ts - t0

# --- plot ---
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    sharex=True,
    constrained_layout=True
)

# Top: DAQ voltage
ax1.plot(f_daq_t, f_daq_v, label="DAQ forward", linewidth=1.5)
ax1.plot(b_daq_t, b_daq_v, label="DAQ backward", linewidth=1.5)
ax1.set_ylabel("Voltage [V]")
ax1.set_title("DAQ and Zaber vs time")
ax1.grid(True, alpha=0.3)
ax1.legend()

# vertical event markers
ax1.axvline(f_event_t, color="tab:blue", linestyle="--", label="forward event time")
ax1.axvline(b_event_t, color="tab:orange", linestyle="--", label="backward event time")


# Bottom: Zaber position
ax2.plot(f_zaber_t, f_zaber_pos, "o-", label="Zaber forward", markersize=4)
ax2.plot(b_zaber_t, b_zaber_pos, "o-", label="Zaber backward", markersize=4)

# Event positions as horizontal lines
ax2.axhline(f_event_z, linestyle="--", linewidth=1.5, label=f"Forward event z = {f_event_z:.3f}")
ax2.axhline(b_event_z, linestyle="--", linewidth=1.5, label=f"Backward event z = {b_event_z:.3f}")

ax2.axvline(f_event_t, color="tab:blue", linestyle="--")
ax2.axvline(b_event_t, color="tab:orange", linestyle="--")

ax2.set_xlabel("Time since t0 [s]")
ax2.set_ylabel("Position")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.show()