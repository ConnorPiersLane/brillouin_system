from dataclasses import dataclass

from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults


@dataclass
class RequestAxialStepScan:
    id: str
    n_measurements: int
    step_size_um: float
    find_reflection_plane: bool | None = None
    eye_tracker_results: EyeTrackerResults | None = None
