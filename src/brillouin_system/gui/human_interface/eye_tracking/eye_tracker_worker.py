# eye_tracker_worker.py
import multiprocessing as mp
import time, traceback
from eye_ipc_shared import ShmRing, ShmFrameSpec
from eye_tracker_dummy import EyeTrackerDummy  # swap with real EyeTracker later

def eye_tracker_worker(req_q: mp.Queue, evt_q: mp.Queue):
    left_ring = right_ring = None
    li = ri = 0
    running = False
    tracker = None
    try:
        while True:
            try:
                cmd = req_q.get_nowait()
            except mp.queues.Empty:
                cmd = None

            if cmd:
                if cmd["type"] == "init":
                    ls = ShmFrameSpec(**cmd["left_spec"])
                    rs = ShmFrameSpec(**cmd["right_spec"])
                    left_ring = ShmRing(ls, create=True)
                    right_ring = ShmRing(rs, create=True)
                    tracker = EyeTrackerDummy(shape=ls.shape)  # <-- init dummy tracker
                    evt_q.put({
                        "type": "inited",
                        "left_spec": ls.__dict__,
                        "right_spec": rs.__dict__
                    })
                elif cmd["type"] == "start":
                    running = True
                    evt_q.put({"type": "started"})
                elif cmd["type"] == "stop":
                    running = False
                    evt_q.put({"type": "stopped"})
                elif cmd["type"] == "shutdown":
                    break

            if running and tracker:
                res = tracker.get_frame()
                left_ring.write_slot(li, res.left)
                right_ring.write_slot(ri, res.right)
                evt_q.put({
                    "type": "frame",
                    "left_idx": li,
                    "right_idx": ri,
                    "pupil": res.pupil,
                    "ts": res.ts
                })
                li = (li + 1) % left_ring.spec.slots
                ri = (ri + 1) % right_ring.spec.slots

    except Exception as e:
        evt_q.put({"type": "error", "msg": str(e), "tb": traceback.format_exc()})
    finally:
        if left_ring:
            left_ring.close(); left_ring.unlink()
        if right_ring:
            right_ring.close(); right_ring.unlink()
