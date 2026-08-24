"""Synthetic dummy camera with a rough two-peak spectrometer simulation.

Serves fully synthetic frames (no data files needed on any machine):

- sample mode -> two water-like Brillouin peaks (shift ~5 GHz)
- reference mode (reference shutter open) -> two EOM sideband peaks whose
  positions follow the dummy microwave's current frequency, so running a
  calibration sweep produces a plausible pixel->GHz mapping
- camera shutter closed -> dark frame (bias + read noise)

Both modes place peaks with the same rough dispersion model (FSR 21.7 GHz,
~294 MHz/px at the window centre, mildly nonlinear across the window), so
an analyzed sample shift comes out near the simulated 5.0 GHz.
"""

import time
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import gaussian_filter1d

from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from .andor_dataclasses import AndorExposure, AndorCameraInfo
from .baseCamera import BaseCamera

# Hardware numbers of the real system (shot/read noise synthesis) — from the
# measured ccd_characteristics, the one home for every obtained camera number.
from brillouin_system.ccd_characteristics import ccd_config as _ccd_config

_GAIN_E_PER_COUNT = _ccd_config.get().sensitivity_e_per_count_preamp_1x
_READ_NOISE_COUNTS = _ccd_config.get().read_noise_counts

# ---- rough spectrometer model ----
_FSR_GHZ = 21.7
_PX_PER_GHZ = 3.4          # ~294 MHz/px at the window centre
_PX_PER_GHZ2 = 0.03        # dispersion varies across the window
_PX_PER_GHZ3 = -0.0015     # small cubic term: calibration is only ROUGHLY polynomial
_WATER_SHIFT_GHZ = 5.0
_WATER_HWHM_GHZ = 0.129
_PSF_SIGMA_PX = 0.8        # instrument line blur applied to every spectrum
_BIAS_COUNTS = 100.0
_REF_EXPOSURE_S = 0.3      # exposure at which the nominal amplitudes apply


class DummyCamera(BaseCamera):
    def __init__(self, microwave=None, shutter_manager=None):
        self.exposure_time = 0.3
        # Conventional mode (emccd gain 0) — a nonzero EM gain would
        # (correctly) disable the photon/Thompson outputs, because the EM
        # sensitivity was never measured.
        self.gain = 0
        self.roi = (0, 160, 0, 20)
        self.binning = (1, 1)
        self.verbose = True
        self._is_streaming = False
        # NEW ATTRIBUTES
        self._flip = False
        self._pre_amp_mode = 16
        self._vss_index = 4
        self._streaming_img_count = 0

        self._microwave = microwave
        self._shutter_manager = shutter_manager
        self._shutter_open = True
        self._rng = np.random.default_rng()

        if self.verbose:
            print("[DummyCamera] initialized")

    def get_camera_info(self):
        model = "Simulated - DummyCamera"
        return {
            "model": model,
            "serial": "DUMMY001",
            "roi": self.get_roi(),
            "binning": self.get_binning(),
            "gain": self.get_emccd_gain(),
            "exposure": self.get_exposure_time(),
            "amp_mode": self.get_amp_mode(),
            "preamp_gain": self.get_preamp_gain(),
            "temperature": "off",
            "flip_image_horizontally": self.get_flip_image_horizontally(),
            "advanced_gain_option": False,
            "vss_speed": self.get_vss_index()
        }

    def get_camera_info_dataclass(self) -> AndorCameraInfo:
        return AndorCameraInfo(**self.get_camera_info())

    def set_from_camera_info(self, info: AndorCameraInfo, do_set_temperature: bool = False):

        if self.verbose:
            print("[DummyCamera] Applying settings from AndorCameraInfo...")

        # Flip
        self.set_flip_image_horizontally(info.flip_image_horizontally)

        # ROI + binning
        x_start, x_end, y_start, y_end = info.roi
        hbin, vbin = info.binning
        self.set_roi(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end)
        self.set_binning(hbin=hbin, vbin=vbin)

        # Exposure + gain
        self.set_exposure_time(info.exposure)
        self.set_emccd_gain(int(info.gain))
        self.set_fixed_pre_amp_mode(info.fixed_pre_amp_mode_index)


        # VSS index
        try:
            self.set_vss_index(int(info.vss_speed))
        except Exception:
            if self.verbose:
                print("[DummyCamera] Could not apply vss_speed; keeping existing VSS index.")

        # Temperature (not simulated)
        if do_set_temperature:
            print(f"[DummyCamera] Temperature requested from info: {info.temperature} (dummy camera ignores this)")

        if self.verbose:
            print("[DummyCamera] Camera state restored from AndorCameraInfo.")

    def get_name(self) -> str:
        return "DummyCamera"

    def open_shutter(self):
        self._shutter_open = True
        print("[DummyCamera] Shutter open")

    def close_shutter(self):
        self._shutter_open = False
        print("[DummyCamera] Shutter closed")

    def _is_reference_mode(self) -> bool:
        if self._shutter_manager is None:
            return False
        return bool(self._shutter_manager.reference._state)

    def snap(self) -> np.ndarray:
        time.sleep(self.exposure_time)

        if not self._shutter_open:
            frame = self._dark_frame()
        elif self._is_reference_mode():
            freq = self._microwave.get_frequency() if self._microwave is not None else 5.75
            # mild EOM/RF-chain roll-off toward high frequency
            amp = 6000.0 * np.exp(-(freq - 4.0) / 8.0)
            frame = self._spectrum_frame(nu_ghz=freq, amp=amp, hwhm_px=0.5)
        else:
            frame = self._spectrum_frame(nu_ghz=_WATER_SHIFT_GHZ, amp=1000.0,
                                         hwhm_px=_WATER_HWHM_GHZ * _PX_PER_GHZ)

        if self._flip:
            frame = np.fliplr(frame)
        return frame

    def _peak_pixels(self, nu_ghz: float, w: int) -> tuple[float, float]:
        """Left/right peak positions for a shift (or EOM sideband) nu_ghz.

        Both peaks sit at optical offsets +-(FSR/2 - nu) from the window
        centre, mapped to pixels with a mildly nonlinear dispersion — the
        pair converges toward the centre as nu approaches FSR/2.
        """
        u = _FSR_GHZ / 2.0 - nu_ghz
        x0 = w / 2.0

        def x_of(uo: float) -> float:
            return x0 + _PX_PER_GHZ * uo + _PX_PER_GHZ2 * uo ** 2 + _PX_PER_GHZ3 * uo ** 3

        return x_of(-u), x_of(+u)

    def _spectrum_frame(self, nu_ghz: float, amp: float, hwhm_px: float) -> np.ndarray:
        h, w = self.get_frame_shape()
        x = np.arange(w, dtype=np.float64)
        x_left, x_right = self._peak_pixels(nu_ghz, w)

        def lorentzian(cen):
            return amp * hwhm_px ** 2 / ((x - cen) ** 2 + hwhm_px ** 2)

        line = lorentzian(x_left) + lorentzian(x_right)
        line = gaussian_filter1d(line, _PSF_SIGMA_PX)
        line *= self.exposure_time / _REF_EXPOSURE_S

        # vertical beam profile across the sline row band
        rows = np.exp(-0.5 * ((np.arange(h) - h / 2.0) / 2.5) ** 2)
        signal = rows[:, None] * line[None, :]

        noise_sigma = np.sqrt(_READ_NOISE_COUNTS ** 2 + signal / _GAIN_E_PER_COUNT)
        frame = _BIAS_COUNTS + signal + self._rng.normal(0.0, 1.0, size=(h, w)) * noise_sigma
        return np.clip(frame, 0, 65535).astype(np.uint16)

    def _dark_frame(self) -> np.ndarray:
        h, w = self.get_frame_shape()
        frame = self._rng.normal(_BIAS_COUNTS, _READ_NOISE_COUNTS, size=(h, w))
        return np.clip(frame, 0, 65535).astype(np.uint16)

    def set_exposure_time(self, seconds: float):
        self.exposure_time = seconds

    def get_exposure_time(self) -> float:
        return self.exposure_time

    def set_emccd_gain(self, gain: float | int):
        self.gain = gain

    def get_emccd_gain(self) -> float | int:
        return self.gain

    def set_roi(self, x_start: int, x_end: int, y_start: int, y_end: int):
        self.roi = (x_start, x_end, y_start, y_end)

    def get_roi(self) -> tuple[int, int, int, int]:
        return self.roi

    def set_binning(self, hbin: int, vbin: int):
        self.binning = (hbin, vbin)

    def get_binning(self) -> tuple[int, int]:
        return self.binning

    def is_opened(self) -> bool:
        return True

    def close(self):
        print("[DummyCamera] Closed.")

    def get_frame_shape(self) -> tuple[int, int]:
        return self.roi[3]-self.roi[2], self.roi[1]-self.roi[0]

    def get_verbose(self) -> bool:
        return self.verbose

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose
        print(f"[DummyCamera] set to self.verbose={self.verbose}")

    def get_preamp_gain(self) -> int:
        return 1

    def get_amp_mode(self) -> str:
        return f"DummyAmpMode(preamp_mode={self._pre_amp_mode})"

    # NEW: Flip image horizontally
    def set_flip_image_horizontally(self, flip: bool):
        self._flip = flip
        if self.verbose:
            print(f"[DummyCamera] Flip image horizontally set to {flip}")

    def get_flip_image_horizontally(self) -> bool:
        return self._flip

    # NEW: Preamp mode
    def set_fixed_pre_amp_mode(self, index: int):
        self._pre_amp_mode = index
        if self.verbose:
            print(f"[DummyCamera] Preamp mode set to index {index}")


    def get_fixed_pre_amp_mode(self) -> int:
        return self._pre_amp_mode

    def get_pre_amp_mode(self) -> int:
        return self._pre_amp_mode

    # NEW: VSS index
    def set_vss_index(self, index: int):
        self._vss_index = index
        if self.verbose:
            print(f"[DummyCamera] VSS index set to {index}")

    def get_vss_index(self) -> int:
        return self._vss_index

    def set_from_config_file(self, config: AndorConfig) -> None:
        if self.verbose:
            print("[DummyCamera] Applying settings from config...")


        self.set_verbose(config.verbose)
        self.set_flip_image_horizontally(config.flip_image_horizontally)

        self.set_roi(
            x_start=config.x_start,
            x_end=config.x_end,
            y_start=config.y_start,
            y_end=config.y_end
        )

        self.set_binning(
            hbin=config.hbin,
            vbin=config.vbin
        )

        self.set_fixed_pre_amp_mode(config.pre_amp_mode)
        self.set_vss_index(config.vss_index)

        print(f'Temperature is {config.temperature}')


        if self.verbose:
            print("[IxonUltra] Configuration applied.")


    def get_exposure_dataclass(self) -> AndorExposure:
        return AndorExposure(
            exposure_time_s=self.get_exposure_time(),
            emccd_gain=self.get_emccd_gain()
        )

    def start_streaming(self, buffer_size: int = 200):
        self._is_streaming = True
        print(f"Started Streaming: buffer {buffer_size}")

    def stop_streaming(self):
        self._is_streaming = False
        print("Ended Streaming")

    def get_newest_streaming_image(self):
        """
        only when streaming
        Return the newest frame available (non-blocking).
        Returns None if no *new* frame since last call.
        """
        if self._streaming_img_count < 100:
            self._streaming_img_count += 1
            return self.snap()[0]
        else:
            self._streaming_img_count = 0
            return 1000 * self.snap()[0]


    @contextmanager
    def streaming(self):
        """
        Safe streaming context manager.
        """
        already_streaming = self._is_streaming
        if not already_streaming:
            self.start_streaming()

        try:
            yield self
        finally:
            if not already_streaming and self._is_streaming:
                self.stop_streaming()