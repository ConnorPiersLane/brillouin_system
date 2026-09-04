"""THE one directory for the instrument's mutable working settings.

Every GUI-editable TOML lives here (fitting, calibration sweep, camera
frames, scanning, sweep scan, eye tracker, tracking, LEDs), so the
instrument's state is one folder to read, diff, and copy — and its git
history documents when the instrument's settings changed.

Deliberately NOT here: measured instrument RECORDS, which live next to
the scripts that produced them (ccd_characteristics/ccd_characteristics
.toml, the laser-offset TOMLs in calibrate_camera_laser_position/) —
records are data with provenance, not settings.

The loader modules (dataclass + load/save + ThreadSafeConfig instance)
stay in their domain packages; only the TOML files are centralised, and
they are loaded LAZILY on first access (LazyThreadSafeConfig), so
importing the library never reads a config file.
"""
from pathlib import Path

CONFIG_DIR = Path(__file__).parent
