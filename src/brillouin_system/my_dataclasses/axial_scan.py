from dataclasses import dataclass

from brillouin_system.calibration.calibration import (
    CalibrationData,
    CalibrationPolyfitParameters,
)
from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults
from brillouin_system.my_dataclasses.measurement_point import MeasurementPoint
from brillouin_system.my_dataclasses.sweep_cycle import SweepCycle
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig
from brillouin_system.scan_managers.sweep_scan_config.sweep_scan_config import SweepScanConfig


@dataclass
class AxialScan:
    i: int  # internal tracker
    id: str
    measurements: list[MeasurementPoint]
    system_state: SystemState
    calibration_params: CalibrationPolyfitParameters | None
    # The RAW calibration frames this scan was taken with, so its own
    # calibration can be re-fitted later (e.g. with a different lineshape
    # model). Analyses must use each scan's own calibration — session-level
    # calibration files drift tens of MHz against the scans — and that is only
    # possible if the frames travel with the scan. ALWAYS stored since
    # 2026-08-24 (the save_calibration_frames off-toggle was removed, user
    # decision); None only for datasets recorded before the field existed or
    # while the old toggle was off.
    calibration_data: CalibrationData | None = None
    eye_tracker_results: EyeTrackerResults | None = None
    reflection_result_forwards: ReflectionResult | None = None
    reflection_result_backwards: ReflectionResult | None = None
    # Set only by the in-out sweep scan: one entry per cycle, in order.
    sweep_cycles: list[SweepCycle] | None = None
    # Acquisition provenance. scanning_config holds the search speed and
    # detection thresholds, which the crossing biases depend on (the ~+5 um
    # direction bias is a 2 mm/s number); sweep_config holds the cycle geometry
    # (approach_um is otherwise only recoverable from the raw Zaber logs).
    sweep_config: SweepScanConfig | None = None
    scanning_config: ScanningConfig | None = None
    # NOTE: an sline_rows field existed 2026-08 only (the acquisition row
    # band) and was REMOVED 2026-08-20 (user decision): the fitter always
    # follows the live config, so storing the band served no purpose. Safe,
    # because calibration and samples share one fitter and hence one band —
    # a common band move is common-mode in pixel space (measured: 2 rows
    # move calibrated shifts 0.1-0.8 MHz per peak, <0.4 MHz on the
    # distance; the ~3-4 MHz/row danger is a cal-vs-sample band MISMATCH,
    # which the shared fitter rules out). Old files carrying the field
    # still load (h5 drops unknown fields, pickles keep it inert).
