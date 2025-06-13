import unittest
import tempfile
import os
import numpy as np
from dataclasses import dataclass

from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    dataclass_to_hdf5_native_dict,
    save_dict_to_hdf5,
    load_dict_from_hdf5,
    dict_to_dataclass_tree,
)

# -----------------------------
# Mock dataclasses for testing
# -----------------------------
@dataclass
class Inner:
    value: int
    label: str

@dataclass
class Outer:
    numbers: list[int]
    inner: Inner
    array: np.ndarray
    flag: bool
    maybe_none: str | None

# -----------------------------
# Known class lookup
# -----------------------------
known_classes = {
    "Inner": Inner,
    "Outer": Outer,
}

# -----------------------------
# Unit Test Suite
# -----------------------------
class TestSafeAndLoadHDF5(unittest.TestCase):

    def setUp(self):
        self.obj = Outer(
            numbers=[1, 2, 3],
            inner=Inner(value=42, label="test"),
            array=np.array([[1.0, 2.0], [3.0, 4.0]]),
            flag=True,
            maybe_none=None
        )

    def roundtrip(self, obj):
        d = dataclass_to_hdf5_native_dict(obj)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "temp.h5")
            save_dict_to_hdf5(path, d)
            return load_dict_from_hdf5(path)

    def test_dataclass_to_hdf5_dict(self):
        d = dataclass_to_hdf5_native_dict(self.obj)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["__dataclass__"], "Outer")
        self.assertEqual(d["inner"]["value"], 42)

    def test_roundtrip_hdf5_preserves_data(self):
        d = self.roundtrip(self.obj)
        self.assertEqual(d["__dataclass__"], "Outer")
        self.assertEqual(d["flag"], True)
        self.assertEqual(d["inner"]["label"], "test")

    def test_full_rehydration(self):
        d = self.roundtrip(self.obj)
        rehydrated = dict_to_dataclass_tree(d, known_classes=known_classes)
        self.assertIsInstance(rehydrated, Outer)
        self.assertEqual(rehydrated.numbers, [1, 2, 3])
        self.assertTrue(np.array_equal(rehydrated.array, self.obj.array))
        self.assertEqual(rehydrated.inner.label, "test")
        self.assertIsNone(rehydrated.maybe_none)

    def test_tuple_roundtrip(self):
        original = (123, "abc", 4.56)
        encoded = dataclass_to_hdf5_native_dict(original)
        decoded = dict_to_dataclass_tree(encoded, known_classes={})
        self.assertEqual(decoded, original)

    def test_object_array_roundtrip(self):
        arr = np.array([{"a": 1}, {"b": 2}], dtype=object)
        encoded = dataclass_to_hdf5_native_dict(arr)
        decoded = dict_to_dataclass_tree(encoded, known_classes={})
        self.assertEqual(decoded.tolist(), [{'a': 1}, {'b': 2}])

    def test_nested_structure(self):
        obj = {
            "meta": {
                "name": "experiment",
                "data": [1, 2, 3]
            },
            "values": np.array([9.0, 8.0])
        }
        encoded = dataclass_to_hdf5_native_dict(obj)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nest.h5")
            save_dict_to_hdf5(path, encoded)
            decoded = load_dict_from_hdf5(path)
        self.assertEqual(decoded["meta"]["name"], "experiment")
        self.assertEqual(decoded["meta"]["data"], [1, 2, 3])
        self.assertEqual(decoded["values"], [9.0, 8.0])

    def test_unsupported_type_raises(self):
        class Unsupported:
            pass
        with self.assertRaises(TypeError):
            dataclass_to_hdf5_native_dict(Unsupported())

if __name__ == "__main__":
    unittest.main()
