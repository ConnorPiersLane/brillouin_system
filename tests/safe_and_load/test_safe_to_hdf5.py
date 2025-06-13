import pickle
import numpy as np
from pathlib import Path

from brillouin_system.saving_and_loading.safe_and_load_dict2hdf5 import (
    dataclass_to_hdf5_native_dict,
    save_dict_to_hdf5,
    load_dict_from_hdf5,
)
from brillouin_system.saving_and_loading.dict2dataclass import dict_to_dataclass_tree
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes

TEST_FILE = "test6.pkl"
HDF5_FILE = "test6.h5"


def deep_equal(a, b, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            print(f"🔴 Mismatch at {path}: different keys {set(a.keys())} != {set(b.keys())}")
            return False
        for key in a:
            if not deep_equal(a[key], b[key], path + f".{key}"):
                return False
        return True

    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print(f"🔴 Mismatch at {path}: list lengths differ ({len(a)} != {len(b)})")
            return False
        for i, (x, y) in enumerate(zip(a, b)):
            if not deep_equal(x, y, path + f"[{i}]"):
                return False
        return True

    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not np.array_equal(a, b):
            print(f"🔴 Mismatch at {path}: arrays differ")
            return False
        return True

    elif isinstance(a, np.ndarray):
        return deep_equal(a.tolist(), b, path)

    elif isinstance(b, np.ndarray):
        return deep_equal(a, b.tolist(), path)

    else:
        if a != b:
            print(f"🔴 Mismatch at {path}: {a} != {b}")
            return False
        return True




def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_conversion_and_hdf5_roundtrip():
    print("🔄 Loading pickle...")
    original_obj = load_pickle(TEST_FILE)

    print("🔁 Converting to dict...")
    dict_version = dataclass_to_hdf5_native_dict(original_obj)

    print("💾 Saving to HDF5...")
    save_dict_to_hdf5(HDF5_FILE, dict_version)

    print("📂 Reloading from HDF5...")
    reloaded_dict = load_dict_from_hdf5(HDF5_FILE)

    print("🔍 Comparing structures...")
    assert deep_equal(dict_version, reloaded_dict), "❌ Mismatch after HDF5 roundtrip!"
    print("✅ Structures match perfectly!")

def test_rehydrate_to_dataclasses():
    print("🔄 Loading pickle again...")
    original_obj = load_pickle(TEST_FILE)

    print("📂 Reloading from HDF5...")
    reloaded_dict = load_dict_from_hdf5(HDF5_FILE)

    print("🧬 Rehydrating into dataclass structure...")
    rehydrated = [
        dict_to_dataclass_tree(item, name_hint="measurement_series", known_classes=known_classes)
        for item in reloaded_dict
    ]

    print("🔍 Comparing original vs rehydrated dataclasses...")

    assert len(original_obj) == len(rehydrated), f"❌ Different list lengths: {len(original_obj)} != {len(rehydrated)}"

    for i, (expected, actual) in enumerate(zip(original_obj, rehydrated)):
        if not deep_equal(dataclass_to_hdf5_native_dict(expected), dataclass_to_hdf5_native_dict(actual), path=f"series[{i}]"):
            raise AssertionError(f"❌ Mismatch in series {i}")

    print("✅ All rehydrated dataclass objects match the original!")




if __name__ == "__main__":
    test_conversion_and_hdf5_roundtrip()
    test_rehydrate_to_dataclasses()
