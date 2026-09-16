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
    # Quality gate for predefined measurements. If set (> 0), the sweep scan is
    # kept only when at least one in-out cycle held below this motion delta
    # (|out-crossing z − in-crossing z|, µm); otherwise the eye moved too much
    # and the scan is treated as failed and NOT saved. None / 0 = no gate
    # (manual sweeps).
    motion_limit_um: float | None = None
