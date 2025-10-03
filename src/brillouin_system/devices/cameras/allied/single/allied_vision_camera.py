from contextlib import ExitStack

from vimba import Vimba, VimbaFeatureError, VimbaCameraError

# If you have these types in your project, keep them; otherwise stub or remove
try:
    from brillouin_system.devices.cameras.allied.allied_config.allied_config import AlliedConfig
except Exception:
    AlliedConfig = None  # Optional: for set_config


class AlliedVisionCamera:
    """
    Single Allied Vision Camera wrapper — tuned for dual-use:
    - Robust initialization across models
    - Transport tuning for max sustainable throughput on 1 GbE
    - Optional inter-packet delay to reduce microbursts
    - Safe streaming callbacks (always re-queue frames)
    """

    def __init__(
        self,
        id: str = "DEV_000F315BC084",
        *,
        pixel_format: str = "Mono8",
        mode: str = "software",
        exposure_us: float = 8000.0,
        throughput_MBps: float = 110.0,    # practical total payload for 1 GbE
        packet_delay: int | None = None,   # e.g., 800 if sharing a link
        enable_ptp: bool = False,
        buffer_count_manual: int = 40,
    ):
        print("[AVCamera] Connecting to Allied Vision Camera…")
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
            print(f"[AVCamera] Camera with ID '{id}' not found. Available:")
            for cam in cameras:
                try:
                    print("  -", cam.get_id())
                except Exception:
                    pass
            raise RuntimeError("[AVCamera] Cannot continue without valid camera.")

        print(f"[AVCamera] …Found camera: {self.camera.get_id()}")

        # Pixel format first (smallest payload by default)
        self.set_pixel_format(pixel_format)

        # Make exposure deterministic before init
        self.set_auto_exposure('Off')

        # One-shot configuration
        self.init(
            mode=mode,
            exposure_us=exposure_us,
            throughput_MBps=throughput_MBps,
            packet_delay=packet_delay,
            enable_ptp=enable_ptp,
            buffer_count_manual=buffer_count_manual,
        )

    # ----------------------------
    # Core init (trigger + transport)
    # ----------------------------
    def init(
        self,
        *,
        mode: str = "software",                 # 'software' | 'freerun' | 'line1'/'hardware'
        exposure_us: float = 8000.0,
        line: str = "Line1",
        activation: str = "RisingEdge",
        debounce_us: int | None = None,
        throughput_MBps: float = 110.0,         # MB/s budget (1 GbE practical max ≈ 105–115 MB/s)
        packet_delay: int | None = None,        # inter-packet delay 'ticks' (try 800 if sharing link)
        enable_ptp: bool = False,
        buffer_count_manual: int = 40,
    ) -> None:
        cam = self.camera

        # 0) Try to stop acquisition if someone left it running
        for feat in ("AcquisitionStop",):
            try:
                cam.get_feature_by_name(feat).run()
            except Exception:
                pass

        def try_set(name: str, value):
            try:
                f = cam.get_feature_by_name(name)
                try:
                    f.set(value)
                    return True
                except Exception:
                    # Fallbacks: bool and common enum mismatches
                    if isinstance(value, str) and value.lower() in ("on", "off"):
                        try:
                            f.set(value.lower() == "on")
                            return True
                        except Exception:
                            pass
                    # Enum conservative fallback if present
                    try:
                        entries = getattr(f, "get_available_entries", lambda: [])()
                        names = [e.get_name() for e in entries]
                        for candidate in ("Basic", "Off", "Disabled", "False"):
                            if candidate in names:
                                f.set(candidate)
                                return True
                    except Exception:
                        pass
            except Exception:
                pass
            return False

        def try_run(name: str):
            try:
                f = cam.get_feature_by_name(name)
                if hasattr(f, "run"):
                    f.run()
            except Exception:
                pass

        # 1) Trigger config: unlatch → set → relatch
        try_set("TriggerSelector", "FrameStart")
        try_set("TriggerMode", "Off")  # unlatch

        if mode == "software":
            try_set("TriggerSource", "Software")
            # Activation not used for software trigger, harmless if set
        elif mode.lower() in ("line1", "hardware", "line"):
            try_set("LineSelector", line)
            try_set("LineMode", "Input")
            if debounce_us is not None:
                try_set("LineDebouncerTime", int(debounce_us))
            try_set("TriggerSource", line)
            try_set("TriggerActivation", activation)
        elif mode == "freerun":
            try_set("TriggerSource", "Software")  # irrelevant; keep TriggerMode Off later
        else:
            raise ValueError(f"Unknown mode: {mode}")

        try_set("AcquisitionMode", "Continuous")

        # 2) Deterministic exposure/gain
        try_set("ExposureAuto", "Off")
        try_set("GainAuto", "Off")
        try:
            self._exposure_feat().set(float(exposure_us))
        except Exception:
            pass

        # Remove FPS clamp if present (names vary by model)
        for feat in ("AcquisitionFrameRateEnable", "AcquisitionFrameRateMode"):
            try_set(feat, "Off")  # bool/enums handled by try_set fallbacks

        # (Re)latch trigger
        if mode == "freerun":
            try_set("TriggerMode", "Off")
        else:
            try_set("TriggerMode", "On")

        # 3) Transport tuning
        try_run("GVSPAdjustPacketSize")  # maximize MTU usage (ensure NIC/switch MTU 9000 for best results)

        # Enable and set throughput cap (bytes/s)
        try_set("DeviceLinkThroughputLimitMode", "On")
        bytes_per_sec = int(throughput_MBps * 1_000_000)  # MB/s → B/s
        for name in ("DeviceLinkThroughputLimit", "StreamBytesPerSecond"):
            try_set(name, bytes_per_sec)

        # Optional pacing to reduce microbursts when sharing a link
        if packet_delay is not None:
            for delay_name in ("GevSCPD", "GevSCPPacketDelay", "GVSPPacketDelay"):
                if try_set(delay_name, int(packet_delay)):
                    break

        # Stream stability niceties
        try_set("StreamBufferHandlingMode", "NewestOnly")
        try_set("StreamBufferCountMode", "Manual")
        try_set("StreamBufferCountManual", int(buffer_count_manual))

        # Optional: PTP for stable timestamps (not required for single cam)
        if enable_ptp:
            try_set("GevIEEE1588", "On")

    # ----------------------------
    # Trigger profiles / helpers
    # ----------------------------
    def set_freerun_mode(self):
        try:
            self.camera.get_feature_by_name('TriggerSelector').set('FrameStart')
            self.camera.get_feature_by_name('TriggerMode').set('Off')
            self.camera.get_feature_by_name('AcquisitionMode').set('Continuous')
            print("[AVCamera] Freerun mode active.")
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

    def fire_software_trigger(self):
        try:
            self.camera.get_feature_by_name('TriggerSoftware').run()
        except Exception as e:
            print(f"[AVCamera] Failed to fire software trigger: {e}")

    # ----------------------------
    # Exposure / Gain / Gamma
    # ----------------------------
    def _exposure_feat(self):
        for name in ("ExposureTime", "ExposureTimeAbs"):
            try:
                return self.camera.get_feature_by_name(name)
            except Exception:
                continue
        raise RuntimeError("[AVCamera] Exposure feature not found.")

    def set_exposure(self, exposure_us: float):
        feat = self._exposure_feat()
        try:
            mn, mx = feat.get_range()
        except Exception:
            mn, mx = None, None
        val = float(exposure_us)
        if mn is not None and mx is not None:
            val = float(max(mn, min(mx, val)))
        feat.set(val)
        print(f"[AVCamera] Exposure set to {val} us")

    def get_exposure(self) -> float:
        return float(self._exposure_feat().get())

    def get_exposure_range(self):
        try:
            return self._exposure_feat().get_range()
        except Exception as e:
            print(f"[AVCamera] Failed to get exposure range: {e}")
            return None

    def get_gain_range(self):
        try:
            feat = self.camera.get_feature_by_name("Gain")
            return feat.get_range()
        except Exception as e:
            print(f"[AVCamera] Failed to get gain range: {e}")
            return None

    def set_gain(self, gain_db: float):
        try:
            feat = self.camera.get_feature_by_name("Gain")
            try:
                mn, mx = feat.get_range()
                gain_db = max(mn, min(mx, float(gain_db)))
            except Exception:
                pass
            feat.set(gain_db)
            print(f"[AVCamera] Gain set to {gain_db} dB")
        except VimbaFeatureError:
            print("[AVCamera] Gain setting not supported or failed.")

    def get_gain(self) -> float:
        return float(self.camera.get_feature_by_name("Gain").get())

    def set_gamma(self, value: float):
        try:
            feat = self.camera.get_feature_by_name("Gamma")
            mn, mx = feat.get_range()
            if not (mn <= value <= mx):
                raise ValueError(f"Gamma must be between {mn} and {mx}")
            feat.set(float(value))
            print(f"[AVCamera] Gamma set to {value}")
        except Exception as e:
            print(f"[AVCamera] Failed to set Gamma: {e}")

    def get_gamma(self) -> float | None:
        try:
            return float(self.camera.get_feature_by_name("Gamma").get())
        except Exception as e:
            print(f"[AVCamera] Failed to get Gamma: {e}")
            return None

    def get_gamma_range(self):
        try:
            feat = self.camera.get_feature_by_name("Gamma")
            return feat.get_range()
        except Exception as e:
            print(f"[AVCamera] Failed to get Gamma range: {e}")
            return None

    # ----------------------------
    # ROI helpers (step-aligned)
    # ----------------------------
    def _align_to_step(self, value, feat):
        try:
            mn, mx = feat.get_range()
            inc = getattr(feat, "get_increment", lambda: 1)()
            v = max(mn, min(mx, int(value)))
            if inc and inc > 1:
                v = mn + ((v - mn) // inc) * inc
            return int(v)
        except Exception:
            return int(value)

    def set_roi(self, OffsetX, OffsetY, Width, Height):
        try:
            w_feat = self.camera.get_feature_by_name("Width")
            h_feat = self.camera.get_feature_by_name("Height")
            ox_feat = self.camera.get_feature_by_name("OffsetX")
            oy_feat = self.camera.get_feature_by_name("OffsetY")

            Width  = self._align_to_step(Width,  w_feat)
            Height = self._align_to_step(Height, h_feat)
            self.camera.get_feature_by_name("Width").set(Width)
            self.camera.get_feature_by_name("Height").set(Height)

            OffsetX = self._align_to_step(OffsetX, ox_feat)
            OffsetY = self._align_to_step(OffsetY, oy_feat)
            self.camera.get_feature_by_name("OffsetX").set(OffsetX)
            self.camera.get_feature_by_name("OffsetY").set(OffsetY)

            print(f"[AVCamera] ROI set to x:{OffsetX} y:{OffsetY} w:{Width} h:{Height}")
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
            w = self.camera.get_feature_by_name("Width").get_range()[1]
            h = self.camera.get_feature_by_name("Height").get_range()[1]
            ox = self.camera.get_feature_by_name("OffsetX").get_range()[0]
            oy = self.camera.get_feature_by_name("OffsetY").get_range()[0]
            return {"OffsetX": ox, "OffsetY": oy, "Width": w, "Height": h}
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to get max ROI: {e}")
            return None

    def set_max_roi(self):
        max_roi = self.get_max_roi()
        if max_roi:
            self.set_roi(max_roi["OffsetX"], max_roi["OffsetY"], max_roi["Width"], max_roi["Height"])
            print("[AVCamera] Set to max ROI.")
        else:
            print("[AVCamera] Could not retrieve max ROI to set.")

    # ----------------------------
    # Pixel format helpers (public enum API)
    # ----------------------------
    def get_available_pixel_formats(self):
        try:
            feat = self.camera.get_feature_by_name("PixelFormat")
            return [e.get_name() for e in feat.get_available_entries()]
        except Exception as e:
            print(f"[AVCamera] Failed to get available pixel formats: {e}")
            return []

    def get_pixel_format(self):
        try:
            return str(self.camera.get_feature_by_name("PixelFormat").get())
        except Exception as e:
            print(f"[AVCamera] Failed to get pixel format: {e}")
            return None

    def set_pixel_format(self, format_str: str):
        try:
            feat = self.camera.get_feature_by_name("PixelFormat")
            feat.set(format_str)
            print(f"[AVCamera] Pixel format set to {format_str}")
        except Exception as e:
            print(f"[AVCamera] Failed to set pixel format: {e}")

    # ----------------------------
    # Streaming
    # ----------------------------
    def start_stream(self, frame_callback, buffer_count: int = 40):
        if self.streaming:
            print("[AVCamera] Already streaming.")
            return
        self._last_callback = frame_callback

        def stream_handler(cam, frame):
            try:
                frame_callback(frame)
            except Exception as e:
                print(f"[AVCamera] Frame callback error: {e}")
            finally:
                cam.queue_frame(frame)

        # Ensure continuous acquisition
        self.set_acquisition_mode("Continuous")
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

    def snap(self, timeout_ms: int = 2000):
        """Capture a single frame; temporarily switch to freerun to guarantee acquisition."""
        was_streaming = self.streaming
        if was_streaming:
            self.stop_stream()
        self.set_freerun_mode()
        frame = self.camera.get_frame(timeout_ms=timeout_ms)
        if was_streaming:
            self.start_stream(self._last_callback)
        return frame

    def set_acquisition_mode(self, mode: str = "Continuous"):
        try:
            self.camera.get_feature_by_name("AcquisitionMode").set(mode)
            print(f"[AVCamera] Acquisition mode set to {mode}")
        except VimbaFeatureError as e:
            print(f"[AVCamera] Failed to set AcquisitionMode to {mode}: {e}")

    # ----------------------------
    # Bulk config apply (optional)
    # ----------------------------
    def set_config(self, cfg: AlliedConfig):
        was_streaming = self.streaming
        if was_streaming:
            self.stop_stream()
        try:
            if hasattr(cfg, 'offset_x'):
                self.set_roi(cfg.offset_x, cfg.offset_y, cfg.width, cfg.height)
            if hasattr(cfg, 'exposure'):
                self.set_exposure(cfg.exposure)
            if hasattr(cfg, 'gain'):
                self.set_gain(cfg.gain)
            if hasattr(cfg, 'gamma'):
                self.set_gamma(cfg.gamma)
            print(f"[AVCamera] Applied full config for {getattr(cfg, 'id', 'unknown')}")
        except Exception as e:
            print(f"[AVCamera] Failed to apply full config: {e}")
        finally:
            if was_streaming:
                self.start_stream(self._last_callback)

    # ----------------------------
    # Lifetime
    # ----------------------------
    def close(self):
        if self.streaming:
            self.stop_stream()
        self.stack.close()
        print("[AVCamera] Camera and Vimba shut down.")

    # Context-manager ergonomics
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # do not suppress exceptions
