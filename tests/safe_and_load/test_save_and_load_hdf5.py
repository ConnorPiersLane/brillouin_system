import unittest
import tempfile
import os

import h5py
import numpy as np
from dataclasses import dataclass, is_dataclass, fields
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    dataclass_to_hdf5_native_dict,
    save_dict_to_hdf5,
    load_dict_from_hdf5,
    dict_to_dataclass_tree,
    register_dataclass,
    known_classes
)

@register_dataclass
@dataclass
class Inner:
    value: int
    label: str

@register_dataclass
@dataclass
class Outer:
    numbers: list[int]
    inner: Inner
    array: np.ndarray
    flag: bool
    maybe_none: str | None

class TestSafeAndLoadHDF5Full(unittest.TestCase):

    def roundtrip_raw(self, obj):
        encoded = dataclass_to_hdf5_native_dict(obj)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "file.h5")
            save_dict_to_hdf5(path, encoded)
            return load_dict_from_hdf5(path)

    def roundtrip_rehydrated(self, obj):
        raw = self.roundtrip_raw(obj)
        return dict_to_dataclass_tree(raw)

    def test_full_dataclass_roundtrip(self):
        obj = Outer(
            numbers=[1, 2, 3],
            inner=Inner(99, "data"),
            array=np.array([[1.5, 2.5], [3.5, 4.5]]),
            flag=False,
            maybe_none=None
        )
        rehydrated = self.roundtrip_rehydrated(obj)
        self.assertIsInstance(rehydrated, Outer)
        self.assertTrue(np.array_equal(rehydrated.array, obj.array))
        self.assertTrue(np.array_equal(rehydrated.numbers, obj.numbers))
        self.assertEqual(rehydrated.inner.label, obj.inner.label)
        self.assertIsNone(rehydrated.maybe_none)

    def test_tuple_roundtrip(self):
        t = (1, "x", 3.14)
        encoded = dataclass_to_hdf5_native_dict(t)
        decoded = dict_to_dataclass_tree(encoded)
        self.assertEqual(decoded, t)

    def test_object_array(self):
        obj_arr = np.array([{"x": 1}, {"y": 2}], dtype=object)
        encoded = dataclass_to_hdf5_native_dict(obj_arr)
        decoded = dict_to_dataclass_tree(encoded)
        self.assertTrue(np.array_equal(decoded, obj_arr))

    def test_primitive_list_roundtrip(self):
        obj = [10, 20, 30]
        out = self.roundtrip_raw(obj)
        self.assertEqual(out.tolist(), obj)

    def test_list_of_strs(self):
        obj = ["a", "b", "c"]
        out = self.roundtrip_raw(obj)
        self.assertEqual(list(out), obj)

    def test_dict_with_ndarray_values(self):
        obj = {"a": np.array([1, 2, 3]), "b": "hello"}
        out = self.roundtrip_raw(obj)
        self.assertTrue(np.array_equal(out["a"], obj["a"]))
        self.assertEqual(out["b"], obj["b"])

    def test_nested_dict_list_combo(self):
        obj = {"nested": [{"x": 1}, {"y": 2}]}
        out = self.roundtrip_raw(obj)
        self.assertEqual(out, obj)

    def test_none_marker_behavior(self):
        self.assertIsNone(dict_to_dataclass_tree("__NONE__"))

    def test_unknown_dataclass_raises(self):
        bad_data = {"__dataclass__": "DoesNotExist", "x": 1}
        with self.assertRaises(ValueError):
            dict_to_dataclass_tree(bad_data)

    def test_non_string_dict_key_raises(self):
        with self.assertRaises(TypeError):
            dataclass_to_hdf5_native_dict({1: "bad"})

    def test_unsupported_type_raises(self):
        class Custom:
            pass
        with self.assertRaises(TypeError):
            dataclass_to_hdf5_native_dict(Custom())

    def test_version_check_fails(self):
        obj = {"a": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad_version.h5")
            with h5py.File(path, 'w') as f:
                f.attrs['__version__'] = '9999.0'
                f.attrs['value'] = 123
            with self.assertRaises(ValueError):
                load_dict_from_hdf5(path)

if __name__ == "__main__":
    unittest.main()
