from dataclasses import asdict

import threading
from copy import deepcopy

# ---------- Thread-safe config wrapper ----------
class ThreadSafeConfig:
    def __init__(self, data_obj):
        self._lock = threading.Lock()
        self._data = data_obj

    def _ensure_loaded(self):
        """Hook for LazyThreadSafeConfig; called under the lock."""
        pass

    def get(self):
        with self._lock:
            self._ensure_loaded()
            return deepcopy(self._data)

    def set(self, field, value):
        with self._lock:
            self._ensure_loaded()
            setattr(self._data, field, value)

    def update(self, **kwargs):
        with self._lock:
            self._ensure_loaded()
            for k, v in kwargs.items():
                setattr(self._data, k, v)

    def get_field(self, field):
        with self._lock:
            self._ensure_loaded()
            return getattr(self._data, field)

    def get_raw(self):  # non-deepcopy for internal save use
        with self._lock:
            self._ensure_loaded()
            return self._data

    def asdict(self):
        with self._lock:
            self._ensure_loaded()
            return asdict(self._data)


class LazyThreadSafeConfig(ThreadSafeConfig):
    """ThreadSafeConfig that loads its data on FIRST access instead of at
    construction, so a module-level config instance does not read its TOML
    at import time. Pass a zero-argument loader; a load error surfaces on
    first use instead of breaking the import of the whole package.
    """

    def __init__(self, loader):
        super().__init__(data_obj=None)
        self._loader = loader

    def _ensure_loaded(self):
        if self._data is None:
            self._data = self._loader()
