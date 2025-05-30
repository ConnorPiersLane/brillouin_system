# import threading
import time
import numpy as np
from pylablib.devices.Andor import AndorSDK2Camera
from .baseCamera import BaseCamera

class IxonUltra(BaseCamera):


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
                 amp_mode_index: int=9,
                 verbose=True):
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
        """

        self.verbose = verbose

        self.cam = AndorSDK2Camera(
            idx=index,
            temperature=temperature,
            fan_mode=fan_mode,
        )
        self.set_amp_mode_by_index(amp_mode_index)
        print("[IxonUltra] Preamp index:", self.cam.get_preamp())
        print("[IxonUltra] Pe", self.get_amp_mode())
        print("[IxonUltra] Preamp gain (e⁻/count):", self.get_preamp_gain())

        if temperature != "off":
            self.cam.set_temperature(temperature, enable_cooler=True)
            self._wait_for_cooling(target_temp=temperature)


        self.set_roi(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end)
        self.set_binning(hbin=hbin, vbin=vbin)
        self.set_emccd_gain_advanced(gain, advanced=advanced_gain_option)
        self.set_exposure_time(seconds=exposure_time)

        self.open_shutter()

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

    def set_emccd_gain(self, gain: float | int):
        #with self._lock:
        if gain == 1:  # Gain==1 not allowed, switch to gain=0 (deactivate gain in this case)
            gain = 0
        actual_gain, adv = self.get_emccd_gain_advanced()
        if adv:
            print("[IxonUltra Camera Warning] Advanced gain mode was enabled but is now disabled!")
        self.cam.set_EMCCD_gain(gain, advanced=False)
        if self.verbose:
            actual_gain, adv = self.get_emccd_gain_advanced()
            print(f"[IxonUltra] Gain set to {actual_gain}, Advanced mode: {adv}")


    def set_emccd_gain_advanced(self, gain: float | int, advanced: bool = False):
        #with self._lock:
            if advanced:
                print("[IxonUltra Camera Warning] Advanced gain mode enabled: sensor damage possible!")
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
            return (hstart, hend, vstart, vend)

    def get_binning(self) -> tuple[int, int]:
        #with self._lock:
            hstart, hend, vstart, vend, hbin, vbin = self.cam.get_roi()
            return (hbin, vbin)

    def snap(self) -> np.ndarray:
        #with self._lock:
            frame = self.cam.snap()
            return frame

    def get_frame_shape(self) -> tuple[int, int]:
        #with self._lock:
            hstart, hend, vstart, vend, hbin, vbin = self.cam.get_roi()
            width = (hend - hstart) // hbin
            height = (vend - vstart) // vbin
            return (height, width)

    def set_amp_mode_by_index(self, index: int):
        modes = self.cam.get_all_amp_modes()
        if index < 0 or index >= len(modes):
            raise ValueError(f"Invalid amp mode index: {index}.")
        mode = modes[index]
        self.cam.set_amp_mode(channel=mode.channel, oamp=mode.oamp, hsspeed=mode.hsspeed, preamp=mode.preamp)
        if self.verbose:
            print(f"[IxonUltra] Amplifier mode set to index {index}:")
            print(f"  Channel: {mode.channel}, Output Amp: {mode.oamp} ({mode.oamp_kind}), "
                  f"HSSpeed: {mode.hsspeed} ({mode.hsspeed_MHz} MHz), Preamp: {mode.preamp} ({mode.preamp_gain} e⁻/count)")

    def get_amp_mode(self) -> object:
        """
        """
        return self.cam.get_amp_mode()


    def list_amp_modes(self):
        modes = self.cam.get_all_amp_modes()
        for i, m in enumerate(modes):
            print(f"[{i}] Channel={m.channel}, BitDepth={m.channel_bitdepth}, OAmp={m.oamp} ({m.oamp_kind}), "
                  f"HSSpeed={m.hsspeed} ({m.hsspeed_MHz} MHz), Preamp={m.preamp} ({m.preamp_gain} e⁻/count)")


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
