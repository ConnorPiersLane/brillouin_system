import time
import threading
from vimba import *

def frame_handler(cam, frame):
    print(f"[{cam.get_id()}] Frame received (ID: {frame.get_id()})")
    cam.queue_frame(frame)

def configure_snap_mode(cam):
    cam.TriggerSelector.set("FrameStart")
    cam.TriggerSource.set("Software")
    cam.TriggerMode.set("On")
    cam.AcquisitionMode.set("Continuous")  # Required even for triggered mode

def trigger_camera(cam):
    cam.get_feature_by_name("TriggerSoftware").run()

def snap_both(cam1, cam2, count=1, delay=1.0):
    for _ in range(count):
        t1 = threading.Thread(target=trigger_camera, args=(cam1,))
        t2 = threading.Thread(target=trigger_camera, args=(cam2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        time.sleep(delay)  # Optional delay between snaps

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

            cam1.start_streaming(frame_handler)
            cam2.start_streaming(frame_handler)

            time.sleep(0.5)  # Let streaming settle

            snap_both(cam1, cam2, count=5, delay=2.0)

            cam1.stop_streaming()
            cam2.stop_streaming()

if __name__ == '__main__':
    main()
