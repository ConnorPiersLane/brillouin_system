from dataclasses import dataclass

from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults


@dataclass
class RequestAxialContScan:
    id: str
    speed_um_s: float
    find_reflection_plane: bool | None = None
    eye_tracker_results: EyeTrackerResults | None = None
