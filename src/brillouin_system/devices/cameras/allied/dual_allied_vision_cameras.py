import threading
import time
import queue
from .allied_vision_camera import AlliedVisionCamera

""" Info: 
Camera 0 (DEV_000F315BDC0C) I/O features:
 - EventAcquisitionRecordTrigger
 - EventAcquisitionRecordTriggerFrameID
 - EventAcquisitionRecordTriggerTimestamp
 - EventFrameTrigger
 - EventFrameTriggerFrameID
 - EventFrameTriggerReady
 - EventFrameTriggerReadyFrameID
 - EventFrameTriggerReadyTimestamp
 - EventFrameTriggerTimestamp
 - EventLine1FallingEdge
 - EventLine1FallingEdgeFrameID
 - EventLine1FallingEdgeTimestamp
 - EventLine1RisingEdge
 - EventLine1RisingEdgeFrameID
 - EventLine1RisingEdgeTimestamp
 - EventLine2FallingEdge
 - EventLine2RisingEdge
 - TriggerActivation
 - TriggerDelayAbs
 - TriggerMode
 - TriggerOverlap
 - TriggerSelector
 - TriggerSoftware
 - TriggerSource
[AVCamera] Camera and Vimba shut down.
[AVCamera] Connecting to Allied Vision Camera...
[AVCamera] ...Found camera: DEV_000F315BC084

Camera 1 (DEV_000F315BC084) I/O features:
 - EventAcquisitionRecordTrigger
 - EventAcquisitionRecordTriggerFrameID
 - EventAcquisitionRecordTriggerTimestamp
 - EventFrameTrigger
 - EventFrameTriggerFrameID
 - EventFrameTriggerReady
 - EventFrameTriggerReadyFrameID
 - EventFrameTriggerReadyTimestamp
 - EventFrameTriggerTimestamp
 - EventLine1FallingEdge
 - EventLine1FallingEdgeFrameID
 - EventLine1FallingEdgeTimestamp
 - EventLine1RisingEdge
 - EventLine1RisingEdgeFrameID
 - EventLine1RisingEdgeTimestamp
 - EventLine2FallingEdge
 - EventLine2RisingEdge
 - EventLine3FallingEdge
 - EventLine3RisingEdge
 - EventLine4FallingEdge
 - EventLine4RisingEdge
 - TriggerActivation
 - TriggerDelayAbs
 - TriggerMode
 - TriggerOverlap
 - TriggerSelector
 - TriggerSoftware
 - TriggerSource

"""


import threading
import time
import queue
from .allied_vision_camera import AlliedVisionCamera


class DualAlliedVisionCameras:
    def __init__(self):
        print("[DualCamera] Initializing two Allied Vision cameras...")
        self.cam0 = AlliedVisionCamera(index=0)
        self.cam1 = AlliedVisionCamera(index=1)

        self.q0 = queue.Queue()
        self.q1 = queue.Queue()
        self.running = False

    def configure_software_triggered(self, mode="Continuous"):
        for cam in [self.cam0, self.cam1]:
            cam.set_acquisition_mode(mode)
            cam.camera.get_feature_by_name("TriggerMode").set("On")
            cam.camera.get_feature_by_name("TriggerSource").set("Software")
        print(f"[DualCamera] Cameras configured for {mode} mode with software trigger.")

    def _frame_callback0(self, frame):
        self.q0.put((time.time(), frame))

    def _frame_callback1(self, frame):
        self.q1.put((time.time(), frame))

    def start_synchronized_stream(self, dual_callback):
        self.configure_software_triggered(mode="Continuous")
        self.cam0.start_stream(self._frame_callback0)
        self.cam1.start_stream(self._frame_callback1)
        self.running = True

        def sync_watcher():
            while self.running:
                try:
                    t0, f0 = self.q0.get(timeout=1)
                    t1, f1 = self.q1.get(timeout=1)
                    dual_callback(f0, f1)
                except queue.Empty:
                    print("[DualCamera] Warning: Frame timeout.")

        self._watcher_thread = threading.Thread(target=sync_watcher, daemon=True)
        self._watcher_thread.start()
        print("[DualCamera] Streaming started with synchronized callbacks.")

    def start_auto_trigger(self, interval_sec=0.1):
        def auto_trigger():
            while self.running:
                self.trigger_both()
                time.sleep(interval_sec)

        self._trigger_thread = threading.Thread(target=auto_trigger, daemon=True)
        self._trigger_thread.start()
        print("[DualCamera] Auto-trigger thread started.")

    def trigger_both(self):
        t0 = threading.Thread(target=lambda: self.cam0.camera.get_feature_by_name("TriggerSoftware").run())
        t1 = threading.Thread(target=lambda: self.cam1.camera.get_feature_by_name("TriggerSoftware").run())
        t0.start()
        t1.start()
        t0.join()
        t1.join()

    def snap_both(self):
        self.configure_software_triggered(mode="SingleFrame")

        image0 = [None]
        image1 = [None]
        barrier = threading.Barrier(2)

        def snap_cam(cam, out):
            barrier.wait()
            cam.camera.get_feature_by_name("TriggerSoftware").run()
            try:
                out[0] = cam.camera.get_frame()
            except Exception as e:
                print(f"[DualCamera] Snap error on {cam}: {e}")
                out[0] = None

        t0 = threading.Thread(target=snap_cam, args=(self.cam0, image0))
        t1 = threading.Thread(target=snap_cam, args=(self.cam1, image1))
        t0.start()
        t1.start()
        t0.join()
        t1.join()

        if image0[0] is None or image1[0] is None:
            print("[DualCamera] Error: One or both snap results are None.")
        return image0[0], image1[0]

    def stop(self):
        print("[DualCamera] Stopping streams...")
        self.running = False
        self.cam0.stop_stream()
        self.cam1.stop_stream()

    def close(self):
        print("[DualCamera] Closing cameras...")
        self.stop()
        self.cam0.close()
        self.cam1.close()


