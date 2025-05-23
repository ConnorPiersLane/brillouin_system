from dataclasses import dataclass, asdict
from pathlib import Path
import tomli
import tomli_w
from copy import deepcopy
import threading


# Path to the TOML configuration file
find_peaks_config_toml_path = Path(__file__).parent.resolve() / "fitting_config.toml"


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
class SelectedRows:
    selected_rows: list[int]


@dataclass
class FindPeaksConfig:
    prominence_fraction: float
    min_peak_width: int
    min_peak_height: int
    rel_height: float
    wlen_pixels: int


# ---------- Load/save helpers ----------

def load_selected_rows(path: Path) -> SelectedRows:
    with path.open("rb") as f:
        data = tomli.load(f)

    if "sline" not in data or "selected_rows" not in data["sline"]:
        raise KeyError("Missing [sline] section or selected_rows key in TOML.")

    return SelectedRows(selected_rows=data["sline"]["selected_rows"])


def save_selected_rows(path: Path, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)

    data["sline"] = {
        "selected_rows": config.get_field("selected_rows")
    }

    with path.open("wb") as f:
        tomli_w.dump(data, f)


def load_find_peaks_config_section(path: Path, section: str) -> FindPeaksConfig:
    with path.open("rb") as f:
        raw = tomli.load(f)["find_peaks"][section]

    converted = {
        k: (None if str(v).lower() == "none" else v)
        for k, v in raw.items()
    }

    return FindPeaksConfig(**converted)


def save_find_peaks_config_section(path: Path, section: str, config: ThreadSafeConfig):
    with path.open("rb") as f:
        data = tomli.load(f)

    if "find_peaks" not in data:
        data["find_peaks"] = {}

    raw_obj = config.get_raw()
    serialized = {
        k: ("None" if v is None else v)
        for k, v in asdict(raw_obj).items()
    }

    data["find_peaks"][section] = serialized

    with path.open("wb") as f:
        tomli_w.dump(data, f)


# ---------- Global config instances ----------

sample_config = ThreadSafeConfig(load_find_peaks_config_section(find_peaks_config_toml_path, "sample"))
reference_config = ThreadSafeConfig(load_find_peaks_config_section(find_peaks_config_toml_path, "reference"))
sline_config = ThreadSafeConfig(load_selected_rows(find_peaks_config_toml_path))
