from dataclasses import dataclass

from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults


@dataclass
class RequestSweepScan:
    id: str
    eye_tracker_results: EyeTrackerResults | None = None
