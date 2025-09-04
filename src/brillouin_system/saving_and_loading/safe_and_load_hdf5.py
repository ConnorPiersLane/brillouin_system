import h5py
import numpy as np
from dataclasses import is_dataclass, fields
from typing import Any, Union

# =============================================================================
# Special Markers used in HDF5 Serialization
# =============================================================================
"""
Markers:
- '__NONE__'      : represents None values
- '__ndarray__'   : represents object-type numpy arrays
- '__dataclass__' : indicates a serialized dataclass, stores the class name
- '__tuple__'     : represents a tuple structure
- '__version__'   : format version identifier
"""

_NONE_MARKER = "__NONE__"
_ARRAY_MARKER = "__ndarray__"
_DATACLASS_MARKER = "__dataclass__"
_VERSION = "1.0"

# =============================================================================
# Dataclass Registry (Auto-Register via Decorator)
# =============================================================================



# =============================================================================
# Serialization Functions
# =============================================================================

def dataclass_to_hdf5_native_dict(obj: Any) -> Union[dict, list, tuple, str, int, float, bool]:
    """Convert a dataclass or supported type into an HDF5-native Python structure."""
    if obj is None:
        return _NONE_MARKER
    elif isinstance(obj, (int, float, bool, str)):
        return obj
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return {_ARRAY_MARKER: True, "items": [dataclass_to_hdf5_native_dict(x) for x in obj]}
        return obj
    elif is_dataclass(obj):
        result = {_DATACLASS_MARKER: obj.__class__.__name__}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = dataclass_to_hdf5_native_dict(value)
        return result
    elif isinstance(obj, list):
        return [dataclass_to_hdf5_native_dict(x) for x in obj]
    elif isinstance(obj, tuple):
        return {"__tuple__": True, "items": [dataclass_to_hdf5_native_dict(x) for x in obj]}
    elif isinstance(obj, dict):
        for k in obj.keys():
            if not isinstance(k, str):
                raise TypeError(f"HDF5 keys must be strings. Got {type(k)}: {k}")
        return {str(k): dataclass_to_hdf5_native_dict(v) for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type for HDF5-native conversion: {type(obj)} (value: {obj})")

def save_dict_to_hdf5(filepath: str, data_dict: dict) -> None:
    """Write a native dictionary structure to an HDF5 file."""
    with h5py.File(filepath, 'w') as f:
        f.attrs['__version__'] = _VERSION
        _write_to_hdf5_group(f, data_dict)

def _write_to_hdf5_group(h5group: h5py.Group, data: Any) -> None:
    """Recursively write native Python types into an HDF5 group."""
    if isinstance(data, dict):
        if data.get("__tuple__") is True:
            h5group.attrs["__tuple__"] = True
            items_group = h5group.create_group("items")
            for idx, item in enumerate(data["items"]):
                _write_to_hdf5_group(items_group.create_group(str(idx)), item)
            return

        if data.get("__ndarray__") is True:
            h5group.attrs["__ndarray__"] = True
            items_group = h5group.create_group("items")
            for idx, item in enumerate(data["items"]):
                _write_to_hdf5_group(items_group.create_group(str(idx)), item)
            return

        for key, val in data.items():
            if key in {"__tuple__", "__ndarray__"}:
                continue  # already handled above as attribute
            if not isinstance(key, str):
                raise TypeError(f"HDF5 requires string keys, got {type(key)}: {key}")
            _write_to_hdf5_group(h5group.create_group(key), val)

    elif isinstance(data, list):
        if all(isinstance(x, (int, float, np.number, bool)) for x in data):
            h5group.create_dataset('value', data=np.array(data))
        elif all(isinstance(x, (int, float, bool, str)) or x == _NONE_MARKER for x in data):
            h5group.attrs['value'] = np.array(data, dtype=object)
        else:
            for idx, item in enumerate(data):
                _write_to_hdf5_group(h5group.create_group(str(idx)), item)

    elif isinstance(data, np.ndarray):
        if data.dtype == object:
            for idx, item in enumerate(data):
                _write_to_hdf5_group(h5group.create_group(str(idx)), item)
        else:
            h5group.create_dataset('value', data=data)

    elif isinstance(data, (int, float, bool, str)):
        h5group.attrs['value'] = data

    elif data == _NONE_MARKER:
        h5group.attrs['value'] = _NONE_MARKER

    else:
        raise TypeError(f"Cannot store unsupported type: {type(data)}")

# =============================================================================
# Deserialization Functions
# =============================================================================

def load_dict_from_hdf5(filepath: str) -> Any:
    """Load an HDF5 file and return a native Python dictionary structure."""
    with h5py.File(filepath, 'r') as f:
        version = f.attrs.get('__version__', '1.0')
        if version != _VERSION:
            raise ValueError(f"Unsupported HDF5 format version: {version}")
        return _read_hdf5_group(f)

def _read_hdf5_group(h5group: h5py.Group) -> Any:
    """Recursively read an HDF5 group back into native Python objects."""

    #print(f"🔍 Entering group: {h5group.name}, keys={list(h5group.keys())}, attrs={dict(h5group.attrs)}")

    if h5group.attrs.get("__tuple__") and "items" in h5group:
        item_keys = sorted(h5group["items"].keys(), key=int)
        return {"__tuple__": True, "items": [_read_hdf5_group(h5group["items"][k]) for k in item_keys]}

    if h5group.attrs.get("__ndarray__") and "items" in h5group:
        item_keys = sorted(h5group["items"].keys(), key=int)
        return {"__ndarray__": True, "items": [_read_hdf5_group(h5group["items"][k]) for k in item_keys]}


    if 'value' in h5group.attrs:
        return h5group.attrs['value']

    if 'value' in h5group and isinstance(h5group.get('value', None), h5py.Dataset):
        val = h5group['value'][()]
        if isinstance(val, np.void):
            return val.tolist()
        if isinstance(val, np.ndarray):
            return val
        return val



    keys = list(h5group.keys())
    is_list = all(k.isdigit() for k in keys)
    keys = sorted(keys, key=lambda k: int(k) if k.isdigit() else k)

    if is_list:
        return [_read_hdf5_group(h5group[k]) for k in keys]
    else:
        return {k: _read_hdf5_group(h5group[k]) for k in keys}

def dict_to_dataclass_tree(data: Any, known: dict[str, Any] = None) -> Any:
    """Recursively convert a deserialized structure back into dataclass instances."""
    if isinstance(data, str) and data == _NONE_MARKER:
        return None

    # Use global known_classes if no override is passed
    resolved_known = known if known is not None else globals().get("known_classes", {})

    if isinstance(data, list):
        return [dict_to_dataclass_tree(item, resolved_known) for item in data]

    if isinstance(data, dict):
        if data.get("__ndarray__") is True and "items" in data:
            return np.array([dict_to_dataclass_tree(item, resolved_known) for item in data["items"]], dtype=object)

        if data.get("__tuple__") is True:
            return tuple(dict_to_dataclass_tree(x, resolved_known) for x in data["items"])

        cls_name = data.get("__dataclass__")
        if cls_name:
            if cls_name not in resolved_known:
                raise ValueError(f"Unknown dataclass '{cls_name}'. Please register it in known_classes.")
            cls = resolved_known[cls_name]
            kwargs = {f.name: dict_to_dataclass_tree(data[f.name], resolved_known)
                      for f in fields(cls) if f.name in data}
            return cls(**kwargs)

        return {k: dict_to_dataclass_tree(v, resolved_known) for k, v in data.items()}

    return data
