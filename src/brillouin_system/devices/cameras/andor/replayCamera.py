"""Replay camera: a DummyCamera that serves real stored frames.

Loads a saved AxialScan pickle (water measurement frames + the raw
calibration frames stored on the scan) and replays them so the GUI in
dummy mode shows real signals:

- sample mode  -> cycles through the stored water frames
- reference mode (reference shutter open) -> serves the stored calibration
  frame nearest to the dummy microwave's current frequency, re-noised with
  fresh shot noise so repeated snaps at one frequency are not identical
- camera shutter closed (dark acquisition) -> dark level + read noise

The frames are served at their stored ROI regardless of the configured
ROI (the stored calibration only makes sense on the stored pixel axis).
"""

import pickle
import time
from pathlib import Path

import numpy as np

from brillouin_system.logging_utils.logging_setup import get_logger
from .dummyCamera import DummyCamera

log = get_logger(__name__)

# Two-peak water measurement (narrow ROI, main Brillouin pair only), 50 degC,
# 500 ms exposure, with a 41-freq calibration (4.0-8.0 GHz, 0.1 steps) stored
# on every scan.
DEFAULT_REPLAY_PKL = (
    Path.home() / "Dropbox (Personal)" / "Boston" / "Data" / "2026-8-6" / "water_50deg.pkl"
)

# Hardware numbers of the real system (used only to re-noise repeated
# calibration frames / synthesize darks) — from the measured
# ccd_characteristics, the one home for every obtained camera number.
from brillouin_system.ccd_characteristics import ccd_config as _ccd_config

_GAIN_E_PER_COUNT = _ccd_config.get().sensitivity_e_per_count_preamp_1x
_READ_NOISE_COUNTS = _ccd_config.get().read_noise_counts


class ReplayCamera(DummyCamera):

    def __init__(self,
                 data_path: Path | str = DEFAULT_REPLAY_PKL,
                 n_sample_frames: int = 100,
                 microwave=None,
                 shutter_manager=None):
        super().__init__()
        self._microwave = microwave
        self._shutter_manager = shutter_manager
        self._shutter_open = True
        self._rng = np.random.default_rng()

        self._sample_frames: np.ndarray | None = None   # (n, h, w)
        self._cal_freqs: np.ndarray | None = None       # sorted, GHz
        self._cal_frames: np.ndarray | None = None      # (n_freq, h, w), same order
        self._dark_level: float = 100.0
        self._i_sample = 0

        self._load_replay_data(Path(data_path), n_sample_frames)

    # ---------------- data loading ----------------
    def _load_replay_data(self, path: Path, n_sample_frames: int):
        if not path.exists():
            log.warning(f"[ReplayCamera] Replay file not found: {path} — "
                        f"falling back to synthetic DummyCamera frames.")
            return
        t0 = time.time()
        with open(path, "rb") as f:
            scans = pickle.load(f)
        scan = scans[0]

        # Concatenate frames across the file's scans (same session) until the
        # requested count is reached; the calibration comes from the first scan.
        frames = [m.frame_andor for s in scans for m in s.measurements]
        frames = frames[:n_sample_frames]
        self._sample_frames = np.stack(frames).astype(np.float64)

        freq_to_frame: dict[float, np.ndarray] = {}
        for mf in scan.calibration_data.measured_freqs:
            # one frame per frequency in the stored scans; keep the first
            freq_to_frame[round(mf.set_freq_ghz, 6)] = (
                mf.cali_meas_points[0].frame.astype(np.float64))
        freqs = np.array(sorted(freq_to_frame))
        self._cal_freqs = freqs
        self._cal_frames = np.stack([freq_to_frame[f] for f in freqs])

        # dark-level estimate for dark frames: low percentile of a real frame
        self._dark_level = float(np.percentile(self._sample_frames[0], 10))

        h, w = self._sample_frames.shape[1:]
        log.info(f"[ReplayCamera] Loaded {len(frames)} sample frames ({h}x{w}) and "
                 f"{len(freqs)} calibration freqs "
                 f"({freqs[0]:.1f}-{freqs[-1]:.1f} GHz) from {path.name} "
                 f"in {time.time() - t0:.1f} s")

    @property
    def has_replay_data(self) -> bool:
        return self._sample_frames is not None

    @property
    def calibration_freqs(self) -> np.ndarray | None:
        """Sorted stored calibration frequencies in GHz (None without data)."""
        return self._cal_freqs

    # ---------------- state hooks ----------------
    def open_shutter(self):
        self._shutter_open = True
        super().open_shutter()

    def close_shutter(self):
        self._shutter_open = False
        super().close_shutter()

    def _is_reference_mode(self) -> bool:
        if self._shutter_manager is None:
            return False
        return bool(self._shutter_manager.reference._state)

    # ---------------- frame serving ----------------
    def snap(self) -> np.ndarray:
        if not self.has_replay_data:
            return super().snap()

        time.sleep(self.exposure_time)

        if not self._shutter_open:
            frame = self._dark_frame()
        elif self._is_reference_mode():
            frame = self._calibration_frame()
        else:
            frame = self._sample_frame()

        if self._flip:
            frame = np.fliplr(frame)
        return frame

    def _dark_frame(self) -> np.ndarray:
        h, w = self.get_frame_shape()
        return self._rng.normal(self._dark_level, _READ_NOISE_COUNTS, size=(h, w))

    def _sample_frame(self) -> np.ndarray:
        frame = self._sample_frames[self._i_sample % len(self._sample_frames)]
        self._i_sample += 1
        return frame.copy()

    def _calibration_frame(self) -> np.ndarray:
        freq = self._microwave.get_frequency() if self._microwave is not None else 5.75
        i = int(np.argmin(np.abs(self._cal_freqs - freq)))
        if abs(self._cal_freqs[i] - freq) > 1e-3:
            log.info(f"[ReplayCamera] No stored calibration frame at "
                     f"{freq:.3f} GHz; serving nearest ({self._cal_freqs[i]:.1f} GHz).")
        frame = self._cal_frames[i]
        # fresh shot noise so repeated snaps at one frequency differ
        shot_sigma = np.sqrt(np.clip(frame - self._dark_level, 0, None) / _GAIN_E_PER_COUNT)
        return frame + self._rng.normal(0.0, 1.0, size=frame.shape) * shot_sigma

    def get_frame_shape(self) -> tuple[int, int]:
        if self.has_replay_data:
            return self._sample_frames.shape[1:]
        return super().get_frame_shape()

    def set_roi(self, x_start: int, x_end: int, y_start: int, y_end: int):
        super().set_roi(x_start, x_end, y_start, y_end)
        if self.has_replay_data and self.verbose:
            print(f"[ReplayCamera] ROI setting stored but ignored — replaying "
                  f"stored frames of shape {self._sample_frames.shape[1:]}")

    def get_name(self) -> str:
        return "ReplayCamera"

    def get_camera_info(self):
        info = super().get_camera_info()
        info["model"] = "Simulated - ReplayCamera (2026-8-6 water 50degC)"
        return info
