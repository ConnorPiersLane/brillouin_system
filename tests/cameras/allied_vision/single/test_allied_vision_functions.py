#!/usr/bin/env python3
"""
Test all AlliedVisionCamera functions.

- ROI get/set/max
- Exposure get/set/min/max
- Gain get/set/min/max
- Gamma get/set/min/max
- Auto exposure modes
- Acquisition modes
- Snap (single frame)
- Start/stop stream

Note: This script does not display images.
"""

import time

from brillouin_system.devices.cameras.allied.single.allied_vision_camera import AlliedVisionCamera


def main():
    # Replace with your actual camera ID
    cam_id = "DEV_000F315BC084"
    cam = AlliedVisionCamera(id=cam_id)

    print("\n=== ROI Tests ===")
    print("Current ROI:", cam.get_roi())
    print("Max ROI:", cam.get_max_roi())
    cam.set_max_roi()
    print("ROI after set_max_roi:", cam.get_roi())

    print("\n=== Exposure Tests ===")
    exp_range = cam.get_exposure_range()
    print("Exposure range:", exp_range)
    if exp_range:
        mid_exp = (exp_range[0] + exp_range[1]) // 2
        cam.set_exposure(mid_exp)
        print("Exposure set to:", cam.get_exposure())

    print("\n=== Gain Tests ===")
    gain_range = cam.get_gain_range()
    print("Gain range:", gain_range)
    if gain_range:
        mid_gain = (gain_range[0] + gain_range[1]) // 2
        cam.set_gain(mid_gain)
        print("Gain set to:", cam.get_gain())

    print("\n=== Gamma Tests ===")
    gamma_range = cam.get_gamma_range()
    print("Gamma range:", gamma_range)
    if gamma_range:
        mid_gamma = (gamma_range[0] + gamma_range[1]) / 2
        cam.set_gamma(mid_gamma)
        print("Gamma set to:", cam.get_gamma())

    print("\n=== Pixel Format Tests ===")
    formats = cam.get_available_pixel_formats()
    print("Available pixel formats:", formats)
    current_format = cam.get_pixel_format()
    print("Current pixel format:", current_format)
    # Try switching to another format if available
    if formats:
        alt_format = formats[0] if formats[0] != current_format else (formats[1] if len(formats) > 1 else None)
        if alt_format:
            cam.set_pixel_format(alt_format)
            print("Pixel format after set:", cam.get_pixel_format())
            # Reset back to original
            cam.set_pixel_format(current_format)
            print("Pixel format reset to:", cam.get_pixel_format())


    print("\n=== Auto Exposure Tests ===")
    for mode in ["Off", "Once", "Continuous"]:
        cam.set_auto_exposure(mode)
        print("Auto exposure mode:", cam.get_auto_exposure())

    print("\n=== Acquisition Mode Tests ===")
    for mode in ["SingleFrame", "Continuous"]:
        cam.set_acquisition_mode(mode)

    print("\n=== Snap Test ===")
    cam.set_software_trigger()
    cam.set_exposure(50000)
    frame = cam.snap()
    print("Captured frame object:", type(frame))

    print("\n=== Streaming Test ===")
    def on_frame(frame):
        print("Stream callback: got frame object", type(frame))
        print(f"Status: {frame.get_status()}")
        # Stop stream after first callback

    cam.start_stream(on_frame, buffer_count=3)

    # Give stream a moment to run
    time.sleep(1)
    cam.stop_stream()

    print("\n=== Closing Camera ===")
    cam.close()
    print("Done.")

if __name__ == "__main__":
    main()
