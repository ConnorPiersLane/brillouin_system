from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
import threading
from copy import deepcopy

# ---------- Thread-safe config wrapper ----------
class ThreadSafeConfig:
    def __init__(self, data_obj):
        self._lock = threading.Lock()
        self._data = data_obj

    def get(self):
        with self._lock:
            return deepcopy(self._data)

    def set(self, field, value):
        with self._lock:
            setattr(self._data, field, value)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._data, k, v)

    def get_field(self, field):
        with self._lock:
            return getattr(self._data, field)

    def get_raw(self):  # non-deepcopy for internal save use
        with self._lock:
            return self._data

    def asdict(self):
        with self._lock:
            return asdict(self._data)

# ---------- Config models ----------

@dataclass
class FindPeaksConfig:
    prominence_fraction: float
    min_peak_width: int
    min_peak_height: int
    rel_height: float
    wlen_pixels: int

@dataclass
class CalibrationConfig:
    n_per_freq: int
    calibration_freqs: list[float]
    reference: str

@dataclass
class AndorFrameSettings:
    selected_rows: list[int]
    n_dark_images: int
    take_dark_image: bool
    n_bg_images: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    vbin: int
    hbin: int
    amp_mode_index: int

#TODO: remove amp_mode_index
# ---------- Load/save helpers ----------

find_peaks_config_toml_path = Path(__file__).parent.resolve() / "config.toml"

def load_find_peaks_config_section(path: Path, section: str) -> FindPeaksConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["find_peaks"][section]
    return FindPeaksConfig(**raw)

def save_find_peaks_config_section(path: Path, section: str, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)

    if "find_peaks" not in data:
        data["find_peaks"] = {}

    data["find_peaks"][section] = asdict(config.get_raw())

    with path.open("wb") as f:
        tomli_w.dump(data, f)

def load_calibration_config(path: Path) -> CalibrationConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["calibration"]

    return CalibrationConfig(
        n_per_freq=raw["n_per_freq"],
        calibration_freqs=raw["calibration_freqs"],
        reference=raw["reference"]
    )

def save_calibration_config(path: Path, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)

    data["calibration"] = asdict(config.get_raw())

    with path.open("wb") as f:
        tomli_w.dump(data, f)

def load_andor_frame_settings(path: Path) -> AndorFrameSettings:
    with path.open("rb") as f:
        raw = tomli.load(f)["andor_frame"]

    return AndorFrameSettings(
        selected_rows=raw["selected_rows"],
        n_dark_images=raw["n_dark_images"],
        take_dark_image=raw["take_dark_image"],
        n_bg_images=raw["n_bg_images"],
        x_start=raw["x_start"],
        x_end=raw["x_end"],
        y_start=raw["y_start"],
        y_end=raw["y_end"],
        vbin=raw["vbin"],
        hbin=raw["hbin"],
        amp_mode_index=raw["amp_mode_index"]
    )

def save_andor_frame_settings(path: Path, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)

    data["andor_frame"] = asdict(config.get_raw())

    with path.open("wb") as f:
        tomli_w.dump(data, f)

# ---------- Global instances ----------
find_peaks_sample_config = ThreadSafeConfig(load_find_peaks_config_section(find_peaks_config_toml_path, "sample"))
find_peaks_reference_config = ThreadSafeConfig(load_find_peaks_config_section(find_peaks_config_toml_path, "reference"))
calibration_config = ThreadSafeConfig(load_calibration_config(find_peaks_config_toml_path))
andor_frame_config = ThreadSafeConfig(load_andor_frame_settings(find_peaks_config_toml_path))
