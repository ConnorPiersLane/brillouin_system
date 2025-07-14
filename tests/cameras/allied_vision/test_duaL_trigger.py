import time
import threading
from vimba import *

# Shared state
frame_cam0 = None
frame_cam1 = None
event_cam0 = threading.Event()
event_cam1 = threading.Event()

def handler0(cam, frame):
    global frame_cam0
    if frame.get_status() == FrameStatus.Complete:
        frame_cam0 = frame
        event_cam0.set()
    cam.queue_frame(frame)

def handler1(cam, frame):
    global frame_cam1
    if frame.get_status() == FrameStatus.Complete:
        frame_cam1 = frame
        event_cam1.set()
    cam.queue_frame(frame)

def trigger(cam, name):
    print(f"[Trigger] {name} triggering at {time.time():.6f}")
    cam.get_feature_by_name("TriggerSoftware").run()

def main():
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        if len(cams) < 2:
            print("Need at least 2 cameras.")
            return

        cam0, cam1 = cams[0], cams[1]

        with cam0, cam1:
            for cam in [cam0, cam1]:
                cam.get_feature_by_name('TriggerSelector').set('FrameStart')
                cam.get_feature_by_name('TriggerMode').set('On')
                cam.get_feature_by_name('TriggerSource').set('Software')
                cam.get_feature_by_name('AcquisitionMode').set('Continuous')

            cam0.start_streaming(handler0)
            cam1.start_streaming(handler1)

            time.sleep(0.2)  # Let the streams settle

            for i in range(2):
                print(f"\n=== Snap {i + 1} ===")

                # Reset events
                event_cam0.clear()
                event_cam1.clear()

                # Trigger both cams in parallel
                t1 = threading.Thread(target=trigger, args=(cam0, "Cam0"))
                t2 = threading.Thread(target=trigger, args=(cam1, "Cam1"))
                t1.start()
                t2.start()
                t1.join()
                t2.join()

                # Wait for both frames
                event_cam0.wait(timeout=2.0)
                event_cam1.wait(timeout=2.0)

                if not (event_cam0.is_set() and event_cam1.is_set()):
                    raise RuntimeError("One or both frames not received.")

                t0 = frame_cam0.get_timestamp()
                t1_ = frame_cam1.get_timestamp()

                print(f"Cam0 Frame ID: {frame_cam0.get_id()} Timestamp: {t0}")
                print(f"Cam1 Frame ID: {frame_cam1.get_id()} Timestamp: {t1_}")
                print(f"Time delta: {abs(t1_ - t0)/1e6:.3f} ms")

            cam0.stop_streaming()
            cam1.stop_streaming()

if __name__ == "__main__":
    main()
