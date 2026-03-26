from typing import Callable

from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults


class CalibRigLaserPosition:

    def __init__(self,
                 ni,
                 zaber_eye_lens,
                 zaber_hi,
                 get_eyetracker_results: Callable[[],EyeTrackerResults],
                 ):

        self.ni = ni
        self.zaber_eye_lens = zaber_eye_lens
        self.zaber_hi = zaber_hi
        self.get_eyetracker_results: Callable[[],EyeTrackerResults] = get_eyetracker_results

    def move_out_until_reflection_plane(self, angle_deg):
        pass

    def binary_search_to_refine(self):
        pass

    def fit_circl
