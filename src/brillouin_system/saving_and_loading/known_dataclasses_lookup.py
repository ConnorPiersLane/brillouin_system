"""Registry of every dataclass the HDF5 loader may reconstruct.

SELF-MAINTAINING: only the ROOT types — the objects that are saved as
top-level files — are listed. Every dataclass reachable from their type
hints is discovered automatically, so adding a field holding a new nested
dataclass needs no edit here. A NEW top-level file format is the only
thing that requires a new entry. The round-trip test
(tests/safe_and_load/test_axial_scan_roundtrip.py) fails if the discovery
ever misses a stored class.
"""
import typing
from dataclasses import fields, is_dataclass

from brillouin_system.calibration.calibration import CalibrationData
from brillouin_system.devices.zaber_engines.zaber_position import ZaberPosition
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
from brillouin_system.eye_tracker.stereo_imaging.calibration_dataclasses import StereoCalibration
from brillouin_system.my_dataclasses.axial_scan import AxialScan
# Legacy-only root: old saved dark files (see my_dataclasses/background_image.py).
from brillouin_system.my_dataclasses.background_image import BackgroundImage
from brillouin_system.my_dataclasses.display_results import DisplayResults
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum

_SAVED_ROOTS = [
    AxialScan,          # scans — the main data format
    CalibrationData,    # standalone calibration files
    FittedSpectrum,
    DisplayResults,
    StereoCalibration,  # eye-tracker stereo calibration files
    EyeTrackerConfig,
    ZaberPosition,
    BackgroundImage,    # legacy dark files
]


def _walk_hint(hint, found: dict) -> None:
    if is_dataclass(hint) and isinstance(hint, type):
        _collect(hint, found)
        return
    for arg in typing.get_args(hint):
        _walk_hint(arg, found)


def _collect(cls, found: dict) -> None:
    if cls.__name__ in found:
        return
    found[cls.__name__] = cls
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        # A hint that cannot be resolved contributes no children; the
        # round-trip test catches anything this would hide.
        hints = {f.name: f.type for f in fields(cls)
                 if not isinstance(f.type, str)}
    for hint in hints.values():
        _walk_hint(hint, found)


known_classes: dict[str, type] = {}
for _root in _SAVED_ROOTS:
    _collect(_root, known_classes)
