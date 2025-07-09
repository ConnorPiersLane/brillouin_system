import time
from vimba import *

def handler(cam, frame):
    print(f'Frame acquired: ID={frame.get_id()}', flush=True)
    cam.queue_frame(frame)

def main():
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        if not cams:
            print("No cameras found.")
            return

        cam = cams[0]

        with cam:
            # Access features properly
            cam.get_feature_by_name('TriggerSelector').set('FrameStart')
            cam.get_feature_by_name('TriggerMode').set('On')
            cam.get_feature_by_name('TriggerSource').set('Software')
            cam.get_feature_by_name('AcquisitionMode').set('Continuous')

            try:
                cam.start_streaming(handler)

                # Give time for the stream to initialize
                time.sleep(1)

                # Trigger 3 frames via software
                for _ in range(3):
                    cam.get_feature_by_name('TriggerSoftware').run()
                    time.sleep(1)

            finally:
                cam.stop_streaming()

if __name__ == '__main__':
    main()
