from contextlib import ExitStack

from vimba import Vimba, VimbaFeatureError, VimbaCameraError

from brillouin_system.devices.cameras.allied.allied_config.allied_config import AlliedConfig
from brillouin_system.devices.cameras.allied.single.base_allied_vision_camera import BaseAlliedVisionCamera

class AlliedVisionCamera(BaseAlliedVisionCamera):
    """
    Allied Vision Camera Test Configuration and Parameter Ranges
    ============================================================

    This test script verifies functionality of the Allied Vision camera
    using the Vimba SDK. The following ranges were observed from the device:

    - ROI (Region of Interest)
      - Current ROI:  OffsetX=0, OffsetY=0, Width=2048, Height=2048
      - Max ROI:      OffsetX=0, OffsetY=0, Width=2048, Height=2048

    - Exposure
      - Range:        1.0 µs  → 153,391,689.0 µs (~153 s)
      - Example set:  500,000.0 µs

    - Gain
      - Range:        0.0 dB  → 26.0 dB
      - Example set:  13.0 dB

    - Gamma
      - Range:        0.25    → 4.0
      - Example set:  2.125

    - Auto Exposure Modes
      - Off
      - Once
      - Continuous

    - Acquisition Modes
      - SingleFrame
      - Continuous

    Notes
    -----
    - Snap (single-frame capture) successfully returns a `vimba.frame.Frame`.
    - Streaming delivers frames via callback. The stream should be stopped
      outside of the callback to avoid `VmbError.InvalidCall`.
    - ROI may be hardware-limited to full sensor size (2048x2048 in this test).
    """

    def __init__(self, id="DEV_000F315BC084", pixel_format='Mono8', mode="software"):
        print("[AVCamera] Connecting to Allied Vision Camera...")
        self.stack = ExitStack()
        self.vimba = self.stack.enter_context(Vimba.get_instance())
        self.camera = None
        self.streaming = False
        self._last_callback = None

        cameras = self.vimba.get_all_cameras()
        if not cameras:
            raise RuntimeError("[AVCamera] No Allied Vision cameras found.")

        try:
            self.camera = self.stack.enter_context(self.vimba.get_camera_by_id(id))
        except VimbaCameraError:
            print(f"[AVCamera] Camera with ID '{id}' not found.")
            print("[AVCamera] Available cameras:")
            for cam in cameras:
                print(f"  - {cam.get_id()}")
            raise RuntimeError("[AVCamera] Cannot continue without valid camera.")

        print(f"[AVCamera] ...Found camera: {self.camera.get_id()}")
        self.set_pixel_format(format_str=pixel_format)
        # Make exposure deterministic; init() will also try-set safely
        self.set_auto_exposure('Off')

        # Configure once, in one place
        self.init(mode=mode, exposure_us=8000)

    def init(self, *, mode="software", exposure_us=8000, line="Line1",
             activation="RisingEdge", debounce_us=None, limit_mbps=120):
        cam = self.camera

        # 0) Make sure we can change features even if someone left it streaming
        for feat in ("AcquisitionStop",):
            try:
                cam.get_feature_by_name(feat).run()
            except Exception:
                pass

        # Small helpers so we don't crash on models that lack a feature
        def try_set(name, value):
            try:
                cam.get_feature_by_name(name).set(value)
            except Exception:
                pass

        def try_run(name):
            try:
                f = cam.get_feature_by_name(name)
                if hasattr(f, "run"):
                    f.run()
            except Exception:
                pass

        # 1) Trigger config (set, unlatch, set deps, relatch) — Viewer-style
        try_set("TriggerSelector", "FrameStart")
        try_set("TriggerMode", "Off")  # unlatch

        if mode == "software":
            try_set("TriggerSource", "Software")
            # TriggerActivation not used for software; harmless if set
        elif mode.lower() in ("line1", "hardware", "line"):
            # Lines / IO
            try_set("LineSelector", line)  # e.g., "Line1"
            try_set("LineMode", "Input")
            if debounce_us is not None:
                try_set("LineDebouncerTime", int(debounce_us))
            # Core trigger
            try_set("TriggerSource", line)  # "Line1"
            try_set("TriggerActivation", activation)  # "RisingEdge" or "FallingEdge"
        elif mode == "freerun":
            try_set("TriggerSource", "Software")  # irrelevant; we'll keep TriggerMode Off later
        else:
            raise ValueError(f"Unknown mode: {mode}")

        try_set("AcquisitionMode", "Continuous")

        # 2) Exposure/gain consistency (Viewer will quietly do this)
        # Disable autos for deterministic triggering
        try_set("ExposureAuto", "Off")
        try_set("GainAuto", "Off")
        # Exposure feature name differs across models: ExposureTimeAbs vs ExposureTime
        for exp_name in ("ExposureTimeAbs", "ExposureTime"):
            try:
                cam.get_feature_by_name(exp_name).set(float(exposure_us))
                break
            except Exception:
                continue

        # If hardware-triggered, remove any frame-rate clamp
        for feat in ("AcquisitionFrameRateEnable", "AcquisitionFrameRateMode"):
            # Turn off the limiter when present, or set to "Basic" if it's a mode enum
            try_set(feat, "Off")  # harmless if it's a bool or enum; try_set guards failures

        # Finally (re)latch
        if mode == "freerun":
            # In freerun keep TriggerMode Off so frames free-run
            try_set("TriggerMode", "Off")
        else:
            try_set("TriggerMode", "On")

        # 3) Transport niceties (Viewer often does this)
        # Adjust packet size if supported
        try_run("GVSPAdjustPacketSize")
        # Some models require enabling the limit mode first
        try_set("DeviceLinkThroughputLimitMode", "On")
        # limit_mbps is in MB/s-ish; SDKs vary on units — this is best-effort.
        bytes_per_sec = int(limit_mbps * 1_000_000)
        for name in ("DeviceLinkThroughputLimit", "StreamBytesPerSecond"):
            try_set(name, bytes_per_sec)

        # A couple of stream stability QoL tweaks (safe no-ops if unsupported)
        try_set("StreamBufferHandlingMode", "NewestOnly")  # avoid backlog on bursty triggers
        try_set("StreamBufferCountMode", "Manual")
        try_set("StreamBufferCountManual", 8)

    def set_freerun_mode(self):
        try:
            self.camera.TriggerSelector.set("FrameStart")
            self.camera.TriggerMode.set("Off")
            self.camera.AcquisitionMode.set("Continuous")
            print("[AVCamera] Camera set to Freerun mode.")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set Freerun mode: {e}")

    def set_software_trigger(self):
        try:
            self.camera.get_feature_by_name('TriggerSelector').set('FrameStart')
            self.camera.get_feature_by_name('TriggerMode').set('On')
            self.camera.get_feature_by_name('TriggerSource').set('Software')
            self.camera.get_feature_by_name('AcquisitionMode').set('Continuous')
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set Software Trigger: {e}")

    def set_exposure(self, exposure_us):
        feat = self.camera.get_feature_by_name("ExposureTimeAbs")
        feat.set(exposure_us)
        print(f"[AVCamera] Exposure time set to {exposure_us} us")

    def get_exposure(self):
        return self.camera.get_feature_by_name("ExposureTimeAbs").get()

    def set_auto_exposure(self, mode='Off'):
        """
        Set the auto exposure mode.
        mode: 'Off', 'Once', or 'Continuous'
        """
        try:
            feat = self.camera.get_feature_by_name("ExposureAuto")
            feat.set(mode)
            print(f"[AVCamera] Auto exposure mode set to {mode}")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set Auto Exposure mode: {e}")

    def get_auto_exposure(self):
        try:
            return str(self.camera.get_feature_by_name("ExposureAuto").get())
        except VimbaFeatureError:
            print("[AVCamera] Auto Exposure feature not available.")
            return None

    def get_exposure_range(self):
        """
        Return (min, max) range of exposure time in microseconds.
        """
        try:
            feat = self.camera.get_feature_by_name("ExposureTimeAbs")
            return feat.get_range()
        except Exception as e:
            print(f"[AVCamera] Failed to get exposure range: {e}")
            return None

    def get_gain_range(self):
        """
        Return (min, max) range of gain in dB.
        """
        try:
            feat = self.camera.get_feature_by_name("Gain")
            return feat.get_range()
        except Exception as e:
            print(f"[AVCamera] Failed to get gain range: {e}")
            return None

    def set_gain(self, gain_db):
        try:
            self.camera.get_feature_by_name("Gain").set(gain_db)
            print(f"[AVCamera] Gain set to {gain_db} dB")
        except VimbaFeatureError:
            print("[AVCamera] Gain setting not supported or failed.")

    def get_gain(self):
        return self.camera.get_feature_by_name("Gain").get()

    def set_roi(self, OffsetX, OffsetY, Width, Height):
        try:
            self.camera.get_feature_by_name("Width").set(Width)
            self.camera.get_feature_by_name("Height").set(Height)
            self.camera.get_feature_by_name("OffsetX").set(OffsetX)
            self.camera.get_feature_by_name("OffsetY").set(OffsetY)
            print(f"[AVCamera] ROI set to x:{OffsetX} y:{OffsetY} width:{Width} height:{Height}")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set ROI: {e}")

    def get_roi(self):
        return {
            "OffsetX": self.camera.get_feature_by_name("OffsetX").get(),
            "OffsetY": self.camera.get_feature_by_name("OffsetY").get(),
            "Width": self.camera.get_feature_by_name("Width").get(),
            "Height": self.camera.get_feature_by_name("Height").get(),
        }

    def get_max_roi(self):
        try:
            width_feat = self.camera.get_feature_by_name("Width")
            height_feat = self.camera.get_feature_by_name("Height")
            offsetx_feat = self.camera.get_feature_by_name("OffsetX")
            offsety_feat = self.camera.get_feature_by_name("OffsetY")

            max_width = width_feat.get_range()[1]  # (min, max)
            max_height = height_feat.get_range()[1]
            min_offset_x = offsetx_feat.get_range()[0]
            min_offset_y = offsety_feat.get_range()[0]

            return {
                "OffsetX": min_offset_x,
                "OffsetY": min_offset_y,
                "Width": max_width,
                "Height": max_height
            }
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to get max ROI: {e}")
            return None

    def set_max_roi(self):
        """Set the camera to use the maximum allowable ROI."""
        max_roi = self.get_max_roi()
        if max_roi:
            self.set_roi(
                OffsetX=max_roi["OffsetX"],
                OffsetY=max_roi["OffsetY"],
                Width=max_roi["Width"],
                Height=max_roi["Height"]
            )
            print("[AVCamera] Set to max ROI.")
        else:
            print("[AVCamera] Could not retrieve max ROI to set.")

    def set_gamma(self, value: float):
        """
        Set gamma correction.
        Typical range: ~0.1 to 4.0 (depends on camera).
        """
        try:
            feat = self.camera.get_feature_by_name("Gamma")
            min_val, max_val = feat.get_range()
            if not (min_val <= value <= max_val):
                raise ValueError(f"Gamma must be between {min_val} and {max_val}")
            feat.set(value)
            print(f"[AVCamera] Gamma set to {value}")
        except Exception as e:
            print(f"[AVCamera] Failed to set Gamma: {e}")

    def get_gamma(self) -> float:
        """
        Get current gamma value.
        """
        try:
            return float(self.camera.get_feature_by_name("Gamma").get())
        except Exception as e:
            print(f"[AVCamera] Failed to get Gamma: {e}")
            return None

    def get_gamma_range(self):
        """
        Return (min, max) range of gamma values.
        """
        try:
            feat = self.camera.get_feature_by_name("Gamma")
            return feat.get_range()
        except Exception as e:
            print(f"[AVCamera] Failed to get Gamma range: {e}")
            return None

    def set_config(self, cfg: AlliedConfig):
        """Apply a full AlliedConfig object to the camera safely."""
        was_streaming = self.streaming
        if was_streaming:
            self.stop_stream()

        try:
            self.set_roi(cfg.offset_x, cfg.offset_y, cfg.width, cfg.height)
            self.set_exposure(cfg.exposure)
            self.set_gain(cfg.gain)
            self.set_gamma(cfg.gamma)
            print(f"[AVCamera] Applied full config for {cfg.id}")
        except Exception as e:
            print(f"[AVCamera] Failed to apply full config: {e}")

        if was_streaming:
            self.start_stream(self._last_callback)

    def snap(self, timeout_ms=2000):
        """Capture a single frame in freerun (continuous) mode."""
        was_streaming = self.streaming
        if was_streaming:
            self.stop_stream()

        # Ensure camera is in freerun
        self.set_freerun_mode()

        # Grab one frame
        frame = self.camera.get_frame(timeout_ms=timeout_ms)

        if was_streaming:
            self.start_stream(self._last_callback)

        return frame

    def set_acquisition_mode(self, mode="Continuous"):
        """
        "SingleFrame" or "Continuous"
        """
        try:
            self.camera.get_feature_by_name("AcquisitionMode").set(mode)
            print(f"[AVCamera] Acquisition mode set to {mode}")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set AcquisitionMode to {mode}: {e}")


    def get_available_pixel_formats(self):
        """
        Return a list of available pixel formats as strings.
        """
        try:
            feat = self.camera.get_feature_by_name("PixelFormat")
            entries = feat._EnumFeature__entries
            return entries
        except Exception as e:
            print(f"[AVCamera] Failed to get available pixel formats: {e}")
            return []

    def get_pixel_format(self):
        """
        Return the current pixel format as a string.
        """
        try:
            return str(self.camera.get_feature_by_name("PixelFormat").get())
        except Exception as e:
            print(f"[AVCamera] Failed to get pixel format: {e}")
            return None

    def set_pixel_format(self, format_str: str):
        """
        Set the pixel format (e.g. 'Mono8', 'Mono12', 'BayerRG8', etc.).
        PixelFormat.Mono8

        Args:
            format_str (str): must be in get_available_pixel_formats()
        """
        try:
            feat = self.camera.get_feature_by_name("PixelFormat")
            feat.set(format_str)
            print(f"[AVCamera] Pixel format set to {format_str}")
        except Exception as e:
            print(f"[AVCamera] Failed to set pixel format: {e}")



    def start_stream(self, frame_callback, buffer_count=5):
        if self.streaming:
            print("[AVCamera] Already streaming.")
            return

        self._last_callback = frame_callback

        # Set to continuous mode
        self.set_acquisition_mode(mode="Continuous")

        def stream_handler(cam, frame):
            frame_callback(frame)
            cam.queue_frame(frame)

        # Allocate and queue initial frames

        self.camera.start_streaming(stream_handler, buffer_count=buffer_count)
        self.streaming = True
        print("[AVCamera] Started streaming.")

    def stop_stream(self):
        if not self.streaming:
            print("[AVCamera] Not currently streaming.")
            return
        self.camera.stop_streaming()
        self.streaming = False
        print("[AVCamera] Stopped streaming.")


    def close(self):
        if self.streaming:
            self.stop_stream()
        self.stack.close()
        print("[AVCamera] Camera and Vimba shut down.")

