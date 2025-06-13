import pickle
import time
import pprint
import numpy as np
from dataclasses import asdict

from brillouin_system.saving_and_loading.safe_and_load_measurement_series import (
    save_measurements_to_hdf5,
    load_measurements_from_hdf5,
)

# --------------------
# Deep Comparison Utility
# --------------------
def deep_equal(a, b, path=""):
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        print(f"[Mismatch] {path}: One is None, the other is not")
        return False

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not np.array_equal(a, b):
            print(f"[Mismatch] {path}: Arrays differ")
            return False
        return True

    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            print(f"[Mismatch] {path}: Dict keys differ")
            return False
        for k in a:
            if not deep_equal(a[k], b[k], path + f".{k}"):
                return False
        return True

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            print(f"[Mismatch] {path}: List/tuple length differ")
            return False
        for i, (x, y) in enumerate(zip(a, b)):
            if not deep_equal(x, y, path + f"[{i}]"):
                return False
        return True

    if isinstance(a, np.generic) and isinstance(b, np.generic):
        if a.item() != b.item():
            print(f"[Mismatch] {path}: Scalar values differ")
            return False
        return True

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not np.array_equal(a, b):
            print(f"[Mismatch] {path}: Arrays differ")
            return False
        return True

    try:
        if a != b:
            print(f"[Mismatch] {path}: Values differ → {a} vs {b}")
            return False
        return True
    except Exception as e:
        print(f"[⚠] {path}: Comparison failed: {type(a)} vs {type(b)} → {e}")
        return False

# --------------------
# Configuration
# --------------------
pkl_path = "test3.pkl"
hdf5_path = "test3.h5"

# --------------------
# Round-Trip Test
# --------------------
start = time.time()
with open(pkl_path, "rb") as f:
    measurement_series_list = pickle.load(f)
print(f"[✓] Loaded {len(measurement_series_list)} series from {pkl_path}")

save_measurements_to_hdf5(hdf5_path, measurement_series_list)
print(f"[✓] Saved to {hdf5_path}")

loaded_from_hdf5 = load_measurements_from_hdf5(hdf5_path)
print(f"[✓] Reloaded {len(loaded_from_hdf5)} series from HDF5")

# --------------------
# Compare Original vs Reloaded
# --------------------
success = True
for i, (orig, loaded) in enumerate(zip(measurement_series_list, loaded_from_hdf5)):
    orig_dict = asdict(orig)
    loaded_dict = asdict(loaded)
    print(f"\n--- Comparing series {i} ---")
    if not deep_equal(orig_dict, loaded_dict, path=f"series[{i}]"):
        print(f"[✗] Mismatch in series {i}")
        success = False
    else:
        print(f"[✓] Series {i} matches")

# --------------------
# Summary
# --------------------
if success:
    print("\n✅ All series match after round-trip.")
else:
    print("\n❌ Discrepancies found.")

print(f"\n⏱ Done in {time.time() - start:.2f}s")
