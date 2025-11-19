import multiprocessing as mp
import threading, queue, traceback

from brillouin_system.eye_tracker.et1_image_acquisition.dual_camera_proxy import DualCameraProxy


def fitting_worker(in_q, out_q, stop_event):
    while not stop_event.is_set():
        try:
            left, right, ts = in_q.get(timeout=0.05)
        except queue.Empty:
            continue

        # Replace with real fitting logic
        result = {"ts": ts, "pupil": (100, 120, 30), "glints": [(10, 20)]}
        out_q.put(result)


class ETWorker:
    def __init__(self, use_dummy=False):
        self.img_acqui_proxy = DualCameraProxy(dtype="uint8", slots=8, use_dummy=use_dummy)
        self.img_acqui_proxy.start()




    def close(self):
        self.img_acqui_proxy.shutdown()


def running_eye_tracker_worker(req_q, evt_q, proxy_kwargs):
    try:
        proxy = DualCameraProxy(**proxy_kwargs)
        proxy.start()

        # Notify proxy we are live
        evt_q.put({"type": "started"})

        frame_q = queue.Queue(maxsize=2)
        stop_event = mp.Event()

        # Reader thread
        def reader():
            while not stop_event.is_set():
                left, right, ts = proxy._get_frames()
                left = left.copy(); right = right.copy()
                if frame_q.full():
                    frame_q.get_nowait()
                frame_q.put_nowait((left, right, ts))

        threading.Thread(target=reader, daemon=True).start()

        # Fitting process
        fit_in = mp.Queue(maxsize=2)
        fit_out = mp.Queue(maxsize=2)
        fitter = mp.Process(target=fitting_worker, args=(fit_in, fit_out, stop_event))
        fitter.start()

        last_result = None

        while not stop_event.is_set():
            # Check for control messages
            while not req_q.empty():
                cmd = req_q.get()
                if cmd["type"] == "shutdown":
                    stop_event.set()
                elif cmd["type"] == "config":
                    # Could pass config down to proxy or fitter
                    pass

            try:
                frame = frame_q.get(timeout=0.05)
            except queue.Empty:
                frame = None

            if frame:
                if fit_in.full():
                    fit_in.get_nowait()  # drop oldest
                fit_in.put_nowait(frame)

            while not fit_out.empty():
                last_result = fit_out.get_nowait()

            if last_result:
                evt_q.put({"type": "result", "data": last_result})

        proxy.shutdown()
        fitter.join()

    except Exception as e:
        evt_q.put({"type": "error", "msg": str(e), "tb": traceback.format_exc()})
