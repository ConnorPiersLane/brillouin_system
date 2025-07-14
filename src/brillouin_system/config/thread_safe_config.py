from dataclasses import asdict

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