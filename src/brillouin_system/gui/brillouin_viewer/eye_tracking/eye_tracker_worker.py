# eye_tracker_worker.py
import multiprocessing as mp
import time, traceback
import numpy as np
from eye_ipc_shared import ShmRing, ShmFrameSpec

def eye_tracker_worker(req_q: mp.Queue, evt_q: mp.Queue):
    """
    req_q: control messages from backend (start, stop, shutdown, config changes)
    evt_q: push-only events to backend (new_frame, errors)
    """
    # for demo we use dummy frames; replace with real Allied Vision capture
    running = False
    pupil_center = (0.0, 0.0)

    left_ring = right_ring = None
    left_idx = right_idx = 0

    try:
        while True:
            # non-blocking check for control messages
            try:
                cmd = req_q.get_nowait()
            except mp.queues.Empty:
                cmd = None

            if cmd:
                typ = cmd.get("type")
                if typ == "init":
                    # allocate rings here based on desired shapes
                    # spec comes from backend so both sides agree
                    left_spec = ShmFrameSpec(**cmd["left_spec"])
                    right_spec = ShmFrameSpec(**cmd["right_spec"])
                    left_ring = ShmRing(left_spec, create=True)
                    right_ring = ShmRing(right_spec, create=True)
                    evt_q.put({"type": "inited",
                               "left_spec": left_spec.__dict__,
                               "right_spec": right_spec.__dict__})
                elif typ == "start":
                    running = True
                    evt_q.put({"type": "started"})
                elif typ == "stop":
                    running = False
                    evt_q.put({"type": "stopped"})
                elif typ == "shutdown":
                    evt_q.put({"type": "bye"})
                    break

            if running and left_ring and right_ring:
                # ---- replace this with real DualCamera grabs ----
                # simulate 640x480 grayscale frames
                # NB: must match the shape/dtype you negotiated in init
                h, w = left_ring.spec.shape[0], left_ring.spec.shape[1]
                left = np.random.randint(0, 255, (h, w), dtype=np.uint8)
                right = np.random.randint(0, 255, (h, w), dtype=np.uint8)
                # fake pupil detection
                pupil_center = (float(w//2), float(h//2))

                left_ring.write_slot(left_idx, left)
                right_ring.write_slot(right_idx, right)

                evt_q.put({
                    "type": "frame",
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "pupil": pupil_center,
                    "ts": time.time()
                })

                left_idx = (left_idx + 1) % left_ring.spec.slots
                right_idx = (right_idx + 1) % right_ring.spec.slots
                time.sleep(0.02)  # ~50 FPS demo

            else:
                time.sleep(0.005)

    except Exception as e:
        evt_q.put({"type": "error", "msg": str(e), "tb": traceback.format_exc()})
    finally:
        if left_ring:
            left_ring.close()
            left_ring.unlink()
        if right_ring:
            right_ring.close()
            right_ring.unlink()
