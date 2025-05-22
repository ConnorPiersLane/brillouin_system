import threading
import time

import numpy as np

from brillouin_system.devices.cameras.andor.baseCamera import BaseCamera
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
from brillouin_system.devices.cameras.allied.base_mako import BaseMakoCamera
from brillouin_system.devices.microwave_device import Microwave, MicrowaveDummy
from brillouin_system.devices.shutter_device import ShutterManager, ShutterManagerDummy
from brillouin_system.devices.zaber_linear_dummy import ZaberLinearDummy
from brillouin_system.my_dataclasses.measurement_data import MeasurementData
from brillouin_system.my_dataclasses.camera_settings import CameraSettings
from brillouin_system.my_dataclasses.fitting_results import FittingResults
from brillouin_system.utils import brillouin_spectrum_fitting
from brillouin_system.devices.zaber_linear import ZaberLinearController


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 using contrast stretching."""
    img = np.copy(image)
    img = np.clip(img, 0, np.percentile(img, 99))  # contrast stretch
    img = 255 * (img / img.max()) if img.max() > 0 else img
    return img.astype(np.uint8)


def select_image_row_for_fitting(frame: np.ndarray, window_size: int) -> tuple[np.ndarray, list]:
    """
    Select the brightest region in the image for Brillouin fitting.
    If window_size is even, chooses the brighter neighbor to center the window.
    Returns the sum of a vertical window of rows centered (or near-centered) on the brightest row.
    """
    row_sums = frame.sum(axis=1)
    center_row = np.argmax(row_sums)

    if window_size % 2 == 1:
        # Odd window: center cleanly
        half_window = window_size // 2
        start = max(center_row - half_window, 0)
        end = min(center_row + half_window + 1, frame.shape[0])
    else:
        # Even window: test both neighbor-centered options
        upper_start = max(center_row - window_size // 2, 0)
        lower_start = max(center_row - window_size // 2 + 1, 0)

        upper_end = min(upper_start + window_size, frame.shape[0])
        lower_end = min(lower_start + window_size, frame.shape[0])

        upper_sum = row_sums[upper_start:upper_end].sum()
        lower_sum = row_sums[lower_start:lower_end].sum()

        if lower_sum > upper_sum:
            start, end = lower_start, lower_end
        else:
            start, end = upper_start, upper_end

    selected_rows = list(range(start, end))
    summed_spectrum = frame[start:end, :].sum(axis=0)

    # sline, rows
    return summed_spectrum, selected_rows


def compute_frequency_shift(fsr: float, sd: float, interpeak_px: float):
    if interpeak_px is np.nan:
        return None
    else:
        return 0.5 * (fsr - interpeak_px * sd)



class BrillouinManager:




    def __init__(self,
                 camera: BaseCamera | DummyCamera,
                 shutter_manager: ShutterManager | ShutterManagerDummy,
                 microwave: Microwave | MicrowaveDummy,
                 zaber: ZaberLinearController | ZaberLinearDummy,
                 mako_camera: BaseMakoCamera,
                 is_sample_illumination_continuous: bool = False,
                 n_pixel_rows: int = 3,
                 sd : float = 0,
                 fsr: float = 0,
                 ):
        # Devices
        self.camera: BaseCamera | DummyCamera = camera
        self.mako_camera: BaseMakoCamera = mako_camera
        self.shutter_manager: ShutterManager | ShutterManagerDummy = shutter_manager
        self.microwave: Microwave | MicrowaveDummy = microwave
        self.zaber = zaber

        # State
        self.is_sample_illumination_continuous: bool = is_sample_illumination_continuous
        self.is_reference_mode: bool = False
        self.do_background_subtraction: bool = False
        self.do_save_images: bool = False


        # Spectrometer Values
        self.sd = sd
        self.fsr = fsr

        self.bg_image = None

        self.init_shutters()

    def init_shutters(self):
        if self.is_reference_mode:
            self.shutter_manager.change_to_reference()
        else:
            self.shutter_manager.change_to_objective()
        if self.is_sample_illumination_continuous:
            self.shutter_manager.sample.open()
        else:
            self.shutter_manager.sample.close()



    # ---------------- Change Modes ----------------

    def change_illumination_mode_to_continuous(self):
        self.is_sample_illumination_continuous = True
        self.shutter_manager.sample.open()
        print("[BrillouinManager] Switched to continuous illumination mode.")

    def change_illumination_mode_to_pulsed(self):
        self.is_sample_illumination_continuous = False
        self.shutter_manager.sample.close()
        print("[BrillouinManager] Switched to pulsed illumination mode.")

    def change_to_reference_mode(self):
        self.is_reference_mode = True
        self.shutter_manager.change_to_reference()
        print("[BrillouinManager] Switched to reference mode.")

    def change_to_sample_mode(self):
        self.is_reference_mode = False
        self.shutter_manager.change_to_objective()
        print("[BrillouinManager] Switched to sample mode.")

    # ----------------- Background Subtraction ----------------- #

    def start_background_subtraction(self):
        if self.is_background_image_available():
            self.do_background_subtraction = True
        else:
            self.do_background_subtraction = False
            print("[BrillouinManager] No Background Image available")

    def stop_background_subtraction(self):
        self.do_background_subtraction = False

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        if not self.is_background_image_available():
            print("[AcquisitionManager] No background image available")
            return frame
        return frame - self.bg_image


    def acquire_background_image(self, number_of_images_to_be_taken: int):
        """Capture and average multiple frames to use as background."""

        if self.is_reference_mode:
            self.shutter_manager.reference.close()
        else:
            if self.is_sample_illumination_continuous:
                self.shutter_manager.sample.close()
            else:
                pass # shutter should already be closed
        time.sleep(0.05)  # Optional delay before acquisition

        if isinstance(self.camera, DummyCamera):
            time.sleep(5)
            self.bg_image = self._get_camera_snap() * 0.99  # simulate something
        else:
            self.bg_image = np.mean(
                np.stack([self._get_camera_snap() for _ in range(number_of_images_to_be_taken)]),
                axis=0
            )
        print("[BrillouinManager] Background Image acquired.")

        if self.is_reference_mode:
            self.shutter_manager.reference.open()
        else:
            if self.is_sample_illumination_continuous:
                self.shutter_manager.sample.open()
            else:
                pass # do not open shutter, we are in snap mode

    def is_background_image_available(self) -> bool:
        return self.bg_image is not None

    # ---------------- Get Frames  ----------------
    def _get_camera_snap(self) -> np.ndarray:
        """Pull a raw frame from the camera."""
        return self.camera.snap().astype(np.float64)


    def snap_and_get_fitting_results(self) -> FittingResults:
        # Get the frame:
        if self.is_reference_mode or self.is_sample_illumination_continuous:
            frame = self._get_camera_snap()
        else:
            frame = self._open_sample_shutter_get_frame_close_shutter(timeout=1)

        if self.do_background_subtraction:
            frame = self.subtract_background(frame)

        # Fit sline
        sline, rows = select_image_row_for_fitting(frame=frame, window_size=3)

        # Fit the spectrum
        interpeak_px, fitted_spect, x_pixels, lorentzian_parameters = brillouin_spectrum_fitting.fitSpectrum(
            np.copy(sline.astype(float)), 1e-4, 1e-4, 50
        )

        x_fit, y_fit = brillouin_spectrum_fitting.refine_fitted_spectrum(x_pixels, lorentzian_parameters, factor=10)

        freq_shift_ghz = compute_frequency_shift(fsr=self.fsr, sd=self.sd, interpeak_px=interpeak_px)

        return FittingResults(
            frame=frame,
            sline=sline,
            used_rows=rows,
            inter_peak_distance_px=interpeak_px,
            fitted_spectrum=fitted_spect,
            x_pixels=x_pixels,
            x_fit_refined = x_fit,
            y_fit_refined = y_fit,
            lorentzian_parameters=lorentzian_parameters,
            sd=self.sd,
            fsr=self.fsr,
            freq_shift_ghz=freq_shift_ghz
        )


    def _open_sample_shutter_get_frame_close_shutter(self, timeout: float) -> np.ndarray:
        """Acquire a frame with temporary shutter open for pulsed mode."""
        frame = None
        timer = threading.Timer(timeout, self.shutter_manager.sample.close)

        try:
            self.shutter_manager.sample.open()
            timer.start()
            frame = self._get_camera_snap()
        finally:
            timer.cancel()
            self.shutter_manager.sample.close()

        return frame

    def get_camera_settings(self) -> CameraSettings:
        return CameraSettings(name=self.camera.get_name(),
                                           exposure_time_s=self.camera.get_exposure_time(),
                                           gain=self.camera.get_gain(),
                                           roi=self.camera.get_roi(),
                                           binning=self.camera.get_binning())



    def get_measurement_data(self,
                             fitting_results: FittingResults) -> MeasurementData:

        if self.do_save_images:
            mako_image = self.mako_camera.snap()
        else:
            mako_image = None
            fitting_results.frame = None


        return MeasurementData(
            fitting_results = fitting_results,
            zaber_position=self.zaber.get_zaber_position_class(),
            camera_settings=self.get_camera_settings(),
            mako_image=mako_image,
            )


