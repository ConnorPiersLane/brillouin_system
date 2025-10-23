# eye_tracker_dummy.py
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class EyeTrackingResults:
    left: np.ndarray
    right: np.ndarray
    pupil: tuple
    ts: float

class EyeTrackerDummy:
    def __init__(self, shape=(240, 320), fps=30):
        self.shape = shape
        self.h, self.w = shape
        self.period = 1.0 / fps
        self._last_ts = 0

    def get_frame(self) -> EyeTrackingResults:
        # throttle to target FPS
        now = time.time()
        delta = now - self._last_ts
        if delta < self.period:
            time.sleep(self.period - delta)
        self._last_ts = time.time()

        # generate random grayscale images
        left = np.random.randint(0, 255, (self.h, self.w), dtype=np.uint8)
        right = np.random.randint(0, 255, (self.h, self.w), dtype=np.uint8)

        # fake pupil at center
        pupil = (self.w // 2, self.h // 2)

        return EyeTrackingResults(left=left, right=right, pupil=pupil, ts=self._last_ts)
