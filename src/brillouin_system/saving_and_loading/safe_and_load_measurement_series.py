import h5py
import numpy as np
import json
from dataclasses import asdict, is_dataclass, fields
from typing import Any, get_origin, get_args, Union

from brillouin_system.my_dataclasses.measurements import MeasurementSeries, MeasurementPoint, MeasurementSettings
from brillouin_system.my_dataclasses.background_image import ImageStatistics
from brillouin_system.my_dataclasses.calibration import CalibrationData
from brillouin_system.my_dataclasses.fitted_results import FittedSpectrum
from brillouin_system.my_dataclasses.camera_settings import CameraSettings
from brillouin_system.my_dataclasses.state_mode import StateMode
from brillouin_system.my_dataclasses.zaber_position import ZaberPosition

# --------------------
# Generic Serialization
# --------------------

def _resolve_type(field_type):
    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin is Union:
        for arg in args:
            if isinstance(arg, type) and is_dataclass(arg):
                return arg
        return next((a for a in args if a is not type(None)), field_type)
    return field_type

def _serialize_dataclass_to_json(obj):
    def convert(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif is_dataclass(val):
            return {f.name: convert(getattr(val, f.name)) for f in fields(val) if getattr(val, f.name) is not None}
        elif isinstance(val, list):
            return [convert(v) for v in val]
        return val
    return json.dumps({f.name: convert(getattr(obj, f.name)) for f in fields(obj) if getattr(obj, f.name) is not None})

def _deserialize_dataclass_from_dict(raw, cls):
    if not is_dataclass(cls):
        raise TypeError(f"Expected a dataclass type, got {cls}")
    kwargs = {}
    for f in fields(cls):
        val = raw.get(f.name, None)
        if val is not None:
            actual_type = _resolve_type(f.type)
            origin = get_origin(actual_type)
            if origin is list:
                subtype = get_args(actual_type)[0]
                if is_dataclass(subtype):
                    val = [_deserialize_dataclass_from_dict(item, subtype) if isinstance(item, dict) else item for item in val]
            elif is_dataclass(actual_type):
                if isinstance(val, dict):
                    val = _deserialize_dataclass_from_dict(val, actual_type)
        kwargs[f.name] = val
    return cls(**kwargs)

def _deserialize_dataclass_from_json(json_str, cls):
    raw = json.loads(json_str)
    return _deserialize_dataclass_from_dict(raw, cls)

# --------------------
# Image Statistics
# --------------------

def _save_image_statistics(h5group, name: str, stats: ImageStatistics | None):
    if stats is None:
        return
    stat_group = h5group.create_group(name)
    for field in fields(ImageStatistics):
        val = getattr(stats, field.name)
        if val is None:
            continue
        if isinstance(val, list):
            list_group = stat_group.create_group(field.name)
            for idx, item in enumerate(val):
                list_group.create_dataset(f"item_{idx}", data=item)
        else:
            stat_group.create_dataset(field.name, data=val)

def _load_image_statistics(h5group, name: str) -> ImageStatistics | None:
    if name not in h5group:
        return None
    group = h5group[name]
    kwargs = {}
    for field in fields(ImageStatistics):
        if field.name in group:
            kwargs[field.name] = group[field.name][:]
        elif field.name in group.keys():
            subgrp = group[field.name]
            kwargs[field.name] = [subgrp[k][:] for k in sorted(subgrp.keys())]
        else:
            kwargs[field.name] = None
    return ImageStatistics(**kwargs)

# --------------------
# CalibrationData
# --------------------

def _save_calibration_data(group, calibration_data: CalibrationData | None):
    if calibration_data is None:
        return
    group.attrs.create("calibration_data", _serialize_dataclass_to_json(calibration_data))

def _load_calibration_data(group) -> CalibrationData | None:
    data_str = group.attrs.get("calibration_data", None)
    if not data_str:
        return None
    if isinstance(data_str, bytes):
        data_str = data_str.decode("utf-8")
    data_dict = json.loads(data_str)
    return _deserialize_dataclass_from_dict(data_dict, CalibrationData)

# --------------------
# Save
# --------------------

def save_measurements_to_hdf5(path: str, measurement_series_list: list[MeasurementSeries]):
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "3.3"

        for i, series in enumerate(measurement_series_list):
            group = f.create_group(f"series_{i}")

            for j, mp in enumerate(series.measurements):
                mg = group.create_group(f"point_{j}")
                for field in fields(mp):
                    val = getattr(mp, field.name)
                    if val is None:
                        continue
                    if isinstance(val, np.ndarray):
                        mg.create_dataset(field.name, data=val)
                    elif is_dataclass(val):
                        mg.attrs.create(field.name, _serialize_dataclass_to_json(val))

            group.attrs.create("settings", _serialize_dataclass_to_json(series.settings))

            sm_group = group.create_group("state_mode")
            for field in fields(series.state_mode):
                val = getattr(series.state_mode, field.name)
                if val is None:
                    continue
                if isinstance(val, ImageStatistics):
                    _save_image_statistics(sm_group, field.name, val)
                elif is_dataclass(val):
                    sm_group.attrs.create(field.name, _serialize_dataclass_to_json(val))
                else:
                    sm_group.attrs.create(field.name, val)

            _save_calibration_data(group, series.calibration_data)

# --------------------
# Load
# --------------------

def load_measurements_from_hdf5(path: str) -> list[MeasurementSeries]:
    result = []
    with h5py.File(path, "r") as f:
        for series_name in f:
            group = f[series_name]

            measurements = []
            for point_name in group:
                if not point_name.startswith("point_"):
                    continue
                mg = group[point_name]
                field_vals = {}
                for field in fields(MeasurementPoint):
                    actual_type = _resolve_type(field.type)
                    if field.name in mg:
                        field_vals[field.name] = mg[field.name][:]
                    elif field.name in mg.attrs:
                        raw_val = mg.attrs[field.name]
                        if isinstance(raw_val, bytes):
                            raw_val = raw_val.decode("utf-8")
                        try:
                            field_vals[field.name] = _deserialize_dataclass_from_json(raw_val, actual_type)
                        except Exception as e:
                            print(f"[Error] Failed to deserialize {field.name} into {field.type}: {e}")
                            field_vals[field.name] = None
                    else:
                        field_vals[field.name] = None
                measurements.append(MeasurementPoint(**field_vals))

            settings_raw = group.attrs.get("settings", "{}")
            if isinstance(settings_raw, bytes):
                settings_raw = settings_raw.decode("utf-8")
            settings = _deserialize_dataclass_from_json(settings_raw, MeasurementSettings)

            sm_group = group["state_mode"]
            sm_vals = {}
            for field in fields(StateMode):
                actual_type = _resolve_type(field.type)
                if field.name in sm_group.attrs:
                    raw_val = sm_group.attrs[field.name]
                    if isinstance(raw_val, bytes):
                        raw_val = raw_val.decode("utf-8")
                    if is_dataclass(actual_type):
                        sm_vals[field.name] = _deserialize_dataclass_from_json(raw_val, actual_type)
                    else:
                        sm_vals[field.name] = raw_val
                elif field.name in sm_group:
                    sm_vals[field.name] = _load_image_statistics(sm_group, field.name)
                else:
                    sm_vals[field.name] = None

            calibration_data = _load_calibration_data(group)

            result.append(MeasurementSeries(
                measurements=measurements,
                settings=settings,
                state_mode=StateMode(**sm_vals),
                calibration_data=calibration_data
            ))

    return result
