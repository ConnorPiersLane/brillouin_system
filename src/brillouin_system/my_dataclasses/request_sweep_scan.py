from dataclasses import dataclass

from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults


@dataclass
class RequestSweepScan:
    id: str
    eye_tracker_results: EyeTrackerResults | None = None
    # Timed mode: ignore SweepScanConfig.n_repeats and instead run as many
    # in-out cycles as fit within SweepScanConfig.max_time_s (which must be
    # > 0). Elapsed time is logged the same as a normal sweep.
    timed: bool = False
