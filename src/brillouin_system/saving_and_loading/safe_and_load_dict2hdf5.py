import inspect
import sys

import h5py
import numpy as np
from dataclasses import is_dataclass, fields

_NONE_MARKER = "__NONE__"


def dataclass_to_hdf5_native_dict(obj):
    if obj is None:
        return _NONE_MARKER
    elif isinstance(obj, (int, float, bool, str)):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, np.generic):
        return obj.item()

    elif is_dataclass(obj):
        return {f.name: dataclass_to_hdf5_native_dict(getattr(obj, f.name)) for f in fields(obj)}
    elif isinstance(obj, tuple):
        return [dataclass_to_hdf5_native_dict(x) for x in obj]
    elif isinstance(obj, list):
        return [dataclass_to_hdf5_native_dict(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): dataclass_to_hdf5_native_dict(v) for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type for HDF5-native conversion: {type(obj)}")


def save_dict_to_hdf5(filepath, data_dict):
    with h5py.File(filepath, 'w') as f:
        _write_to_hdf5_group(f, data_dict)


def _write_to_hdf5_group(h5group, data):
    if isinstance(data, dict):
        for key, val in data.items():
            _write_to_hdf5_group(h5group.create_group(str(key)), val)

    elif isinstance(data, list):
        # If it's a list of only numbers, store as array
        if all(isinstance(x, (int, float, np.number, bool)) for x in data):
            h5group.create_dataset('value', data=np.array(data))
        elif all(isinstance(x, (int, float, bool, str)) or x == _NONE_MARKER for x in data):
            # Handle small scalar lists
            h5group.attrs['value'] = np.array(data, dtype=object)
        else:
            for idx, item in enumerate(data):
                _write_to_hdf5_group(h5group.create_group(str(idx)), item)

    elif isinstance(data, np.ndarray):
        h5group.create_dataset('value', data=data)

    elif isinstance(data, (int, float, bool, str)):
        h5group.attrs['value'] = data

    elif data == _NONE_MARKER:
        h5group.attrs['value'] = _NONE_MARKER

    else:
        raise TypeError(f"Cannot store unsupported type: {type(data)}")


def load_dict_from_hdf5(filepath):
    with h5py.File(filepath, 'r') as f:
        return _read_hdf5_group(f)


def _read_hdf5_group(h5group):
    if 'value' in h5group.attrs:
        return h5group.attrs['value']

    if 'value' in h5group:
        val = h5group['value'][()]
        if isinstance(val, np.void):  # for object arrays
            return val.tolist()
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    keys = sorted(h5group.keys(), key=lambda k: int(k) if k.isdigit() else k)
    is_list = all(k.isdigit() for k in keys)

    result = []
    result_dict = {}
    for key in keys:
        item = _read_hdf5_group(h5group[key])
        if is_list:
            result.append(item)
        else:
            result_dict[key] = item

    return result if is_list else result_dict

