"""LazyThreadSafeConfig: the loader runs on FIRST access, never at
construction — the contract that keeps importing the library from reading
config files.
"""
from dataclasses import dataclass

from brillouin_system.helpers.thread_safe_config import LazyThreadSafeConfig


@dataclass
class DummyConfig:
    value: int
    name: str = "x"


class CountingLoader:
    def __init__(self):
        self.calls = 0

    def __call__(self) -> DummyConfig:
        self.calls += 1
        return DummyConfig(value=42)


def test_loader_does_not_run_at_construction():
    loader = CountingLoader()
    LazyThreadSafeConfig(loader)
    assert loader.calls == 0


def test_loader_runs_once_on_first_access():
    loader = CountingLoader()
    cfg = LazyThreadSafeConfig(loader)

    assert cfg.get().value == 42
    assert cfg.get_field("name") == "x"
    assert cfg.asdict()["value"] == 42
    assert loader.calls == 1


def test_update_before_first_get_loads_then_applies():
    loader = CountingLoader()
    cfg = LazyThreadSafeConfig(loader)

    cfg.update(value=7)
    assert loader.calls == 1
    assert cfg.get().value == 7


def test_get_returns_a_copy():
    cfg = LazyThreadSafeConfig(CountingLoader())
    snapshot = cfg.get()
    snapshot.value = -1
    assert cfg.get().value == 42
