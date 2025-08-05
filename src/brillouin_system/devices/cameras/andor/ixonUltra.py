import time
import numpy as np
from collections import namedtuple

from pylablib.devices.Andor import AndorSDK2Camera
from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from .andor_dataclasses import AndorCameraInfo, AndorExposure
from .baseCamera import BaseCamera



# Define the mode object (match your TAmpModeFull if needed)
AmpMode = namedtuple("AmpMode", ["channel", "oamp", "hsspeed", "preamp"])




class IxonUltra(BaseCamera):
    # Fixed index-to-mode mapping (your reference set)
    FIXED_AMP_MODE_LOOKUP = {
        0: AmpMode(0, 0, 0, 0),
        1: AmpMode(0, 0, 0, 1),
        2: AmpMode(0, 0, 0, 2),
        3: AmpMode(0, 0, 1, 0),
        4: AmpMode(0, 0, 1, 1),
        5: AmpMode(0, 0, 1, 2),
        6: AmpMode(0, 0, 2, 0),
        7: AmpMode(0, 0, 2, 1),
        8: AmpMode(0, 0, 2, 2),
        9: AmpMode(0, 0, 3, 0),
        10: AmpMode(0, 0, 3, 1),
        11: AmpMode(0, 0, 3, 2),
        12: AmpMode(0, 1, 0, 0),
        13: AmpMode(0, 1, 0, 1),
        14: AmpMode(0, 1, 0, 2),
        15: AmpMode(0, 1, 1, 0),
        16: AmpMode(0, 1, 1, 1),
        17: AmpMode(0, 1, 1, 2),
        18: AmpMode(0, 1, 2, 0),
        19: AmpMode(0, 1, 2, 1),
        20: AmpMode(0, 1, 2, 2),
    }
    """
    [0]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=0 (17.0 MHz),       Preamp=0 (1.0 e⁻/count)
    [1]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=0 (17.0 MHz),       Preamp=1 (2.0 e⁻/count)
    [2]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=0 (17.0 MHz),       Preamp=2 (3.0 e⁻/count)
    [3]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=1 (10.0 MHz),       Preamp=0 (1.0 e⁻/count)
    [4]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=1 (10.0 MHz),       Preamp=1 (2.0 e⁻/count)
    [5]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=1 (10.0 MHz),       Preamp=2 (3.0 e⁻/count)
    [6]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=2 (5.0 MHz),        Preamp=0 (1.0 e⁻/count)
    [7]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=2 (5.0 MHz),        Preamp=1 (2.0 e⁻/count)
    [8]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=2 (5.0 MHz),        Preamp=2 (3.0 e⁻/count)
    [9]  Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=3 (1.0 MHz),        Preamp=0 (1.0 e⁻/count)
    [10] Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=3 (1.0 MHz),        Preamp=1 (2.0 e⁻/count)
    [11] Channel=0, BitDepth=16, OAmp=0 (Electron Multiplying), HSSpeed=3 (1.0 MHz),        Preamp=2 (3.0 e⁻/count)
    [12] Channel=0, BitDepth=16, OAmp=1 (Conventional),          HSSpeed=0 (3.0 MHz),        Preamp=0 (1.0 e⁻/count)
    [13] Channel=0, BitDepth=16, OAmp=1 (Conventional),          HSSpeed=0 (3.0 MHz),        Preamp=1 (2.0 e⁻/count)
    [14] Channel=0, BitDepth=16, OAmp=1 (Conventional),          HSSpeed=0 (3.0 MHz),        Preamp=2 (3.0 e⁻/count)
    [15] Channel=0, BitDepth=16, OAmp=1 (Conventional),          HSSpeed=1 (1.0 MHz),        Preamp=0 (1.0 e⁻/count)
    [16] Channel=0, BitDepth=16, OAmp=1 (Conventional),          HSSpeed=1 (1.0 MHz),        Preamp=1 (2.0 e⁻/count)
    [17] Channel=0, BitDepth=16, OAmp=1 (Conventional),          HSSpeed=1 (1.0 MHz),        Preamp=2 (3.0 e⁻/count)
    [18] Channel=0, BitDepth=16, OAmp=1 (Conventional),          HSSpeed=2 (0.08 MHz),       Preamp=0 (1.0 e⁻/count)
    [19] Channel=0, BitDepth=16, OAmp=1 (Conventional),          HSSpeed=2 (0.08 MHz),       Preamp=1 (2.0 e⁻/count)
    [20] Channel=0, BitDepth=16, OAmp=1 (Conventional),          HSSpeed=2 (0.08 MHz),       Preamp=2 (3.0 e⁻/count)
    """

    def __init__(self,
                 index: int = 0,
                 temperature: float | str = "off",
                 fan_mode: str = "full",
                 x_start: int = 1, x_end: int = 512,
                 y_start: int = 1, y_end: int = 512,
                 vbin: int = 1, hbin: int = 1,
                 exposure_time = 0.21,
                 gain: int = 0,
                 advanced_gain_option: bool = False,
                 amp_mode_index: int=18,
                 verbose: bool =True,
                 flip_image_horizontally: bool = False):
        """
        Initialize the IxonUltra camera.

        Args:
            index (int): Camera index. Default is 0.
            temperature (float | str): Cooling temperature in Celsius (-80 to 0) or "off". Default is "off".
            fan_mode (str): Fan mode, either "full", "low", or "off". Default is "on".
            x_start (int): Horizontal ROI start pixel (>= 1).
            x_end (int): Horizontal ROI end pixel (sensor width, e.g., 1024).
            y_start (int): Vertical ROI start pixel (>= 1).
            y_end (int): Vertical ROI end pixel (sensor height, e.g., 1024).
            vbin (int): Vertical binning factor (>= 1). Default is 1.
            hbin (int): Horizontal binning factor (>= 1). Default is 1.
            gain (int): EMCCD gain (0 to 300 typically).
            advanced_gain_option (bool): Use advanced gain (>300). Default is False.
            amp_mode_index: 0-20
        """

        # TDeviceInfo(controller_model='USB', head_model='DU897_BV', serial_number=9303)
        # Controller Model: USB
        # Camera Name(Head Model): DU897_BV
        # Serial Number: 9303

        self.verbose: bool = verbose

        self.flip_image_horizontally: bool = flip_image_horizontally

        self._advanced_gain = advanced_gain_option

        self.cam = AndorSDK2Camera(
            idx=index,
            temperature=temperature,
            fan_mode=fan_mode,
        )
        # Set shift speed:
        desired_speed_index = 4  # Slowest shift speed (3.3 μs)
        # Available VSSpeeds: [0.30000001192092896, 0.5, 0.8999999761581421, 1.7000000476837158, 3.299999952316284]
        self.cam.set_vsspeed(desired_speed_index)
        print(f"[IxonUltra] VSSpeed Index: {self.cam.get_vsspeed()}")
        print(f"[IxonUltra] VSSpeed Period: {self.cam.get_vsspeed_period():.2f} μs")

        self.set_pre_amp_mode(amp_mode_index)
        print("[IxonUltra] Preamp index:", self.cam.get_preamp())
        print("[IxonUltra] Preamp mode", self.get_amp_mode())
        print("[IxonUltra] Preamp gain (e⁻/count):", self.get_preamp_gain())

        if temperature != "off":
            self.cam.set_temperature(temperature, enable_cooler=True)
            self._wait_for_cooling(target_temp=temperature)

        self.set_roi(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end)
        self.set_binning(hbin=hbin, vbin=vbin)
        self.set_emccd_gain(gain, advanced=advanced_gain_option)
        self.set_exposure_time(seconds=exposure_time)

        self.open_shutter()

    def get_camera_info(self):
        info = self.cam.get_device_info()
        model = f"{info.controller_model} - {info.head_model}"
        return {
            "model": model,
            "serial": info.serial_number,
            "roi": self.get_roi(),
            "binning": self.get_binning(),
            "gain": self.get_emccd_gain(),
            "exposure": self.get_exposure_time(),
            "amp_mode": str(self.get_amp_mode()),
            "preamp_gain": self.get_preamp_gain(),
            "temperature": self.cam.get_temperature(),
            "flip_image_horizontally": self.get_flip_image_horizontally(),
            "advanced_gain_option": self._advanced_gain,
            "vss_speed": self.get_vss_index()
        }

    def get_camera_info_dataclass(self) -> AndorCameraInfo:
        return AndorCameraInfo(**self.get_camera_info())

    def get_exposure_dataclass(self) -> AndorExposure:
        return AndorExposure(
            exposure_time_s=self.get_exposure_time(),
            emccd_gain=self.get_emccd_gain()
        )


    def get_name(self) -> str:
        return "IxonUltra"

    def open_shutter(self):
        self.cam.setup_shutter("open")

    def close_shutter(self):
        self.cam.setup_shutter("closed")

    def get_verbose(self) -> bool:
        return self.verbose

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    def _wait_for_cooling(self, target_temp: int | float=0, timeout: int=600):
        start = time.time()
        while time.time() - start < timeout:
            status = self.cam.get_temperature_status()
            temp = self.cam.get_temperature()
            print(f"[IxonUltra] Cooling... Status: {status}, Temp: {temp:.2f} °C")
            # if status == "stabilized":
            #     print("[IxonUltra] Cooling stabilized.")
            #     return
            if temp < target_temp + 3:
                print("[IxonUltra] Target Temperature reached")
                return
            time.sleep(10)
        print("[IxonUltra] Warning: cooling did not stabilize within timeout.")

    def _wait_for_warmup(self, target_temp=0, timeout=1200):
        start = time.time()
        while time.time() - start < timeout:
            temp = self.cam.get_temperature()
            print(f"[IxonUltra] Warming... Temp: {temp:.2f} °C")
            if temp >= target_temp - 0.5:
                return
            time.sleep(10)
        print("[IxonUltra] Warning: warm-up did not complete within timeout.")

    def set_exposure_time(self, seconds: float):
        #with self._lock:
            self.cam.set_exposure(seconds)
            if self.verbose:
                print(f"[IxonUltra] Exposure time set to {self.get_exposure_time():.4f} seconds")

    def set_emccd_gain(self, gain: float | int, advanced: bool = None):
        """Set EMCCD gain. Only applies when in EM mode. Advanced mode is risky!"""
        if advanced is None:
            advanced = self._advanced_gain

        current_mode = self.get_amp_mode()
        if current_mode.oamp == 1:
            if self.verbose:
                print("[IxonUltra Warning] EMCCD gain ignored in Conventional mode (oamp=1).")
            gain = 0

        if gain == 1:
            gain = 0  # Avoid invalid value

        if advanced:
            print("[IxonUltra Warning] Advanced gain mode enabled — sensor risk!")

        self.cam.set_EMCCD_gain(gain, advanced=advanced)

        if self.verbose:
            actual_gain, adv = self.get_emccd_gain_advanced()
            print(f"[IxonUltra] Gain set to {actual_gain}, Advanced mode: {adv}")

    def set_roi(self, x_start: int, x_end: int, y_start: int, y_end: int):
        #with self._lock:
            self.cam.set_roi(hstart=x_start, hend=x_end, vstart=y_start, vend=y_end)
            if self.verbose:
                actual_roi = self.get_roi()
                print(
                    f"[IxonUltra] ROI set to x:({actual_roi[0]}-{actual_roi[1]}), y:({actual_roi[2]}-{actual_roi[3]})")

    def set_binning(self, hbin: int, vbin: int):
        #with self._lock:
            hstart, hend, vstart, vend, _, _ = self.cam.get_roi()
            self.cam.set_roi(hstart=hstart, hend=hend, vstart=vstart, vend=vend, hbin=hbin, vbin=vbin)
            if self.verbose:
                actual_binning = self.get_binning()
                print(f"[IxonUltra] Binning set to hbin={actual_binning[0]}, vbin={actual_binning[1]}")

    def get_exposure_time(self) -> float:
        #with self._lock:
            return self.cam.get_exposure()

    def get_emccd_gain(self) -> int:
        #with self._lock:
        current_mode = self.get_amp_mode()
        if current_mode.oamp == 1:
            if self.verbose:
                print("[IxonUltra Warning] No Gain used in Conventional mode (oamp=1).")
            return 0
        else:
            return self.cam.get_EMCCD_gain()[0]

    def get_preamp_gain(self) -> float:
        """Preamp gain (e⁻/count)"""
        return self.cam.get_preamp_gain()

    def get_emccd_gain_advanced(self) -> tuple[int, bool]:
        #with self._lock:
            return self.cam.get_EMCCD_gain()

    def get_roi(self) -> tuple[int, int, int, int]:
        #with self._lock:
            hstart, hend, vstart, vend, hbin, vbin = self.cam.get_roi()
            return hstart, hend, vstart, vend

    def get_binning(self) -> tuple[int, int]:
        #with self._lock:
            hstart, hend, vstart, vend, hbin, vbin = self.cam.get_roi()
            return hbin, vbin

    def snap(self) -> np.ndarray:
        #with self._lock:
            frame = self.cam.snap()
            if self.flip_image_horizontally:
                frame = np.fliplr(frame)
            return frame

    def get_frame_shape(self) -> tuple[int, int]:
        #with self._lock:
            hstart, hend, vstart, vend, hbin, vbin = self.cam.get_roi()
            width = (hend - hstart) // hbin
            height = (vend - vstart) // vbin
            return height, width



    # ---- Flip image horizontally ----
    def set_flip_image_horizontally(self, flip: bool):
        """Set whether to flip the frame horizontally."""
        self.flip_image_horizontally = flip
        if self.verbose:
            print(f"[IxonUltra] Flip image horizontally set to {flip}")

    def get_flip_image_horizontally(self) -> bool:
        """Return whether horizontal flipping is enabled."""
        return self.flip_image_horizontally


    # ---- Preamp mode index ----
    def set_pre_amp_mode(self, index: int):
        if index not in self.FIXED_AMP_MODE_LOOKUP:
            raise ValueError(f"Invalid amp mode index: {index}")

        mode = self.FIXED_AMP_MODE_LOOKUP[index]

        # Check against available modes
        available_modes = [
            (m.channel, m.oamp, m.hsspeed, m.preamp)
            for m in self.cam.get_all_amp_modes()
        ]

        if (mode.channel, mode.oamp, mode.hsspeed, mode.preamp) not in available_modes:
            print(f"[IxonUltra Warning] Requested amp mode {index} ({mode}) is not listed in available amp modes.")

        # Proceed anyway
        self.cam.set_amp_mode(
            channel=mode.channel,
            oamp=mode.oamp,
            hsspeed=mode.hsspeed,
            preamp=mode.preamp
        )

        if self.verbose:
            print(f"[IxonUltra] Amp Mode set to index {index}: {self.get_amp_mode()}")

    def get_amp_mode(self) -> object:
        """
        Return the current amplifier mode settings object.
        """
        return self.cam.get_amp_mode()

    def list_amp_modes(self):
        modes = self.cam.get_all_amp_modes()
        for i, m in enumerate(modes):
            print(f"[{i}] Channel={m.channel}, BitDepth={m.channel_bitdepth}, OAmp={m.oamp} ({m.oamp_kind}), "
                  f"HSSpeed={m.hsspeed} ({m.hsspeed_MHz} MHz), Preamp={m.preamp} ({m.preamp_gain} e⁻/count)")

    def list_fixed_amp_modes(self):
        for idx, mode in self.FIXED_AMP_MODE_LOOKUP.items():
            oamp_kind = "EM" if mode.oamp == 0 else "Conventional"
            hsspeed_freq = {0: 17.0, 1: 10.0, 2: 5.0, 3: 1.0, 4: 0.08}.get(mode.hsspeed, "?")
            preamp_gain = {0: 1.0, 1: 2.0, 2: 3.0}.get(mode.preamp, "?")
            print(f"[{idx}] {oamp_kind}, {hsspeed_freq} MHz, {preamp_gain} e⁻/count")

    def get_pre_amp_mode(self) -> int:
        """Return current amplifier mode index."""
        # You must track this manually if the SDK does not expose it directly
        # OR return a match index from the current mode
        current = self.cam.get_amp_mode()
        all_modes = self.cam.get_all_amp_modes()
        for i, mode in enumerate(all_modes):
            if (mode.channel == current.channel and
                    mode.oamp == current.oamp and
                    mode.hsspeed == current.hsspeed and
                    mode.preamp == current.preamp):
                return i
        return -1  # not found

    # ---- Vertical shift speed index ----
    def set_vss_index(self, index: int):
        """Set vertical shift speed index."""
        self.cam.set_vsspeed(index)
        if self.verbose:
            print(f"[IxonUltra] VSSpeed index set to {index} (period: {self.cam.get_vsspeed_period():.2f} μs)")

    def get_vss_index(self) -> int:
        """Return current vertical shift speed index."""
        return self.cam.get_vsspeed()

    def set_from_config_file(self, config: AndorConfig) -> None:
        if self.verbose:
            print("[IxonUltra] Applying settings from config...")

        self._advanced_gain = config.advanced_gain_option

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

        self.set_pre_amp_mode(config.pre_amp_mode)
        self.set_vss_index(config.vss_index)

        if config.temperature == "off":
            if not self.cam.get_temperature() > 0:
                print("[IxonUltra] Warming up to 0°C before turning off cooler...")
                self.cam.set_temperature(0, enable_cooler=True)
                self._wait_for_warmup(target_temp=0)
                print("[IxonUltra] Turning off cooler...")
                self.cam.set_cooler(False)
            else:
                self.cam.set_cooler(False)
        else:
            self.cam.set_temperature(config.temperature, enable_cooler=True)
            self._wait_for_cooling(target_temp=config.temperature)


        if self.verbose:
            print("[IxonUltra] Configuration applied.")

    def is_opened(self) -> bool:
        return self.cam is not None and self.cam.is_opened()

    def close(self):
        #with self._lock:
            if hasattr(self, "cam") and self.cam is not None and self.cam.is_opened():
                try:
                    if not self.cam.get_temperature() > 0:
                        print("[IxonUltra] Warming up to 0°C before shutdown...")
                        self.cam.set_temperature(0, enable_cooler=True)
                        self._wait_for_warmup(target_temp=0)
                        print("[IxonUltra] Turning off cooler...")
                        self.cam.set_cooler(False)
                except Exception as e:
                    print(f"[IxonUltra] Error during warmup shutdown: {e}")
                self.close_shutter()
                self.cam.close()
                self.cam = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if hasattr(self, "cam"):
            self.close()
