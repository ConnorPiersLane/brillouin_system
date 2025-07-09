import queue
import threading
import time
from vimba import *

# Queues to store frames from each camera
frame_q1 = queue.Queue()
frame_q2 = queue.Queue()

def handler1(cam, frame):
    print(f"[{cam.get_id()}] Frame received (ID: {frame.get_id()})")
    frame_q1.put(frame)
    cam.queue_frame(frame)

def handler2(cam, frame):
    print(f"[{cam.get_id()}] Frame received (ID: {frame.get_id()})")
    frame_q2.put(frame)
    cam.queue_frame(frame)

def configure_snap_mode(cam):
    cam.TriggerSelector.set("FrameStart")
    cam.TriggerSource.set("Software")
    cam.TriggerMode.set("On")
    cam.AcquisitionMode.set("Continuous")
    cam.get_feature_by_name("ExposureTimeAbs").set(5000)  # 5 ms exposure for fast triggering
    try:
        cam.get_feature_by_name("ExposureAuto").set("Off")
        cam.get_feature_by_name("GainAuto").set("Off")
    except VimbaFeatureError:
        pass  # not all cameras support these features

def trigger_both(cam1, cam2):
    t1 = threading.Thread(target=lambda: cam1.get_feature_by_name("TriggerSoftware").run())
    t2 = threading.Thread(target=lambda: cam2.get_feature_by_name("TriggerSoftware").run())
    t1.start(); t2.start(); t1.join(); t2.join()

def snap_both_return_frames(cam1, cam2):
    # Clear old frames from queues
    while not frame_q1.empty(): frame_q1.get_nowait()
    while not frame_q2.empty(): frame_q2.get_nowait()

    cam1.start_streaming(handler1)
    cam2.start_streaming(handler2)
    time.sleep(0.1)  # Let streaming settle

    trigger_both(cam1, cam2)

    # Block until we get one frame from each camera
    try:
        f1 = frame_q1.get(timeout=5.0)
        f2 = frame_q2.get(timeout=5.0)
    finally:
        cam1.stop_streaming()
        cam2.stop_streaming()

    return f1, f2

def main():
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        if len(cams) < 2:
            print("Need at least 2 cameras.")
            return

        cam1, cam2 = cams[0], cams[1]
        with cam1, cam2:
            configure_snap_mode(cam1)
            configure_snap_mode(cam2)

            print("Snapping both cameras...")
            frame1, frame2 = snap_both_return_frames(cam1, cam2)

            print("Frame 1:")
            print(f"  ID: {frame1.get_id()}")
            print(f"  Timestamp: {frame1.get_timestamp()}")
            print("Frame 2:")
            print(f"  ID: {frame2.get_id()}")
            print(f"  Timestamp: {frame2.get_timestamp()}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("General Error:", e)
