# eye_tracker_proxy.py
import multiprocessing as mp
import uuid
from typing import Optional
from eye_ipc_shared import ShmRing, ShmFrameSpec

class EyeTrackerProxy:
    def __init__(self, frame_shape=(480, 640), dtype='uint8', slots=8):
        # Defer creating Queues/Process until start() to avoid Windows import issues
        self.req_q: Optional[mp.Queue] = None
        self.evt_q: Optional[mp.Queue] = None
        self.proc: Optional[mp.Process] = None


        # Local state
        self.left_ring: Optional[ShmRing] = None
        self.right_ring: Optional[ShmRing] = None
        self.inited = False
        self.running = False

        # Desired spec (we request these; worker allocates)
        base = f"eye_{uuid.uuid4().hex[:8]}"
        self.left_spec = ShmFrameSpec(name=f"{base}_L", shape=frame_shape, dtype=dtype, slots=slots)
        self.right_spec = ShmFrameSpec(name=f"{base}_R", shape=frame_shape, dtype=dtype, slots=slots)

    # proper instance method (NOT nested inside __init__)
    def _make_process(self) -> mp.Process:
        # import here so child process can import this module cleanly
        from eye_tracker_worker import eye_tracker_worker
        # create queues here (not at import / __init__ time)
        self.req_q = mp.Queue()
        self.evt_q = mp.Queue()
        return mp.Process(target=eye_tracker_worker, args=(self.req_q, self.evt_q), daemon=True)

    def start(self):
        if self.proc is None or not self.proc.is_alive():
            self.proc = self._make_process()
            self.proc.start()

        # tell worker to allocate shared memory
        self.req_q.put({
            "type": "init",
            "left_spec": self.left_spec.__dict__,
            "right_spec": self.right_spec.__dict__,
        })

        # wait for 'inited'
        evt = self._wait_for("inited")
        # attach to SM segments the worker just created
        ls = ShmFrameSpec(**evt["left_spec"])
        rs = ShmFrameSpec(**evt["right_spec"])
        self.left_ring = ShmRing(ls, create=False)
        self.right_ring = ShmRing(rs, create=False)
        self.inited = True

    def _wait_for(self, typ):
        while True:
            msg = self.evt_q.get()
            if msg.get("type") == typ:
                return msg
            if msg.get("type") == "error":
                raise RuntimeError(f"Eye worker error: {msg['msg']}\n{msg.get('tb','')}")

    def begin_stream(self):
        if not self.inited:
            self.start()
        self.req_q.put({"type": "start"})
        self._wait_for("started")
        self.running = True

    def end_stream(self):
        if not self.inited:
            return
        self.req_q.put({"type": "stop"})
        self._wait_for("stopped")
        self.running = False

    def poll_frames(self):
        """
        Drain events and yield ready frames (as numpy copies) + metadata.
        Returns a list of dicts: {"left": np.ndarray, "right": np.ndarray, "pupil": (x,y), "ts": float}
        """
        out = []
        if not (self.left_ring and self.right_ring and self.evt_q):
            return out

        # drain quickly to keep latency low
        while not self.evt_q.empty():
            msg = self.evt_q.get_nowait()
            typ = msg.get("type")
            if typ == "frame":
                li = msg["left_idx"]; ri = msg["right_idx"]
                left = self.left_ring.read_slot(li)
                right = self.right_ring.read_slot(ri)
                out.append({"left": left, "right": right, "pupil": msg["pupil"], "ts": msg["ts"]})
            elif typ == "error":
                raise RuntimeError(f"Eye worker error: {msg['msg']}\n{msg.get('tb','')}")
            elif typ in ("started","stopped","inited","bye"):
                pass  # ignore in poll mode
        return out

    def shutdown(self):
        try:
            if self.running:
                self.end_stream()
            if self.req_q:
                self.req_q.put({"type": "shutdown"})
            if self.proc:
                self.proc.join(timeout=2)
        finally:
            if self.left_ring:
                self.left_ring.close()
                self.left_ring = None
            if self.right_ring:
                self.right_ring.close()
                self.right_ring = None
