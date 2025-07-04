import time
from your_module.dual_allied_vision_cameras import DualAlliedVisionCameras  # Update path as needed

def dual_callback(frame0, frame1):
    print("Received synchronized frames:")
    print(f" - Cam 0 shape: {frame0.shape}")
    print(f" - Cam 1 shape: {frame1.shape}")

def main():
    dual_cam = DualAlliedVisionCameras()

    try:
        print("\n--- TEST: Snap Both ---")
        img0, img1 = dual_cam.snap_both()
        print("Snap result:")
        print(f" - Cam 0 shape: {img0.shape}")
        print(f" - Cam 1 shape: {img1.shape}")

        print("\n--- TEST: Synchronized Streaming ---")
        dual_cam.start_synchronized_stream(dual_callback)
        dual_cam.start_auto_trigger(interval_sec=0.2)

        time.sleep(2.5)  # Wait to collect ~10 frames
        dual_cam.stop()

    finally:
        dual_cam.close()

if __name__ == "__main__":
    main()
