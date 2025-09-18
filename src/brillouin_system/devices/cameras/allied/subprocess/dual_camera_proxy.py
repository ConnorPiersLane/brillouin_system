# dual_camera_proxy.py
import multiprocessing as mp
from brillouin_system.helpers.frame_ipc_shared import ShmFrameSpec, ShmRing

class DualCameraProxy:
    def __init__(self, dtype="uint8", slots=8, use_dummy=False):
        self.req_q = mp.Queue()
        self.evt_q = mp.Queue(maxsize=2)
        self.proc = None
        self.left_ring = None
        self.right_ring = None
        self.use_dummy = use_dummy

        base = "dual_" + mp.current_process().name
        self.left_spec  = ShmFrameSpec(f"{base}_L", (1, 1), dtype, slots)   # worker overrides shape
        self.right_spec = ShmFrameSpec(f"{base}_R", (1, 1), dtype, slots)

    def _make_process(self):
        from dual_camera_worker import dual_camera_worker
        return mp.Process(target=dual_camera_worker, args=(self.req_q, self.evt_q), daemon=True)

    def start(self):
        if self.proc is None or not self.proc.is_alive():
            self.proc = self._make_process()
            self.proc.start()

        self.req_q.put({
            "type": "init",
            "left_spec": self.left_spec.__dict__,
            "right_spec": self.right_spec.__dict__,
            "use_dummy": self.use_dummy,
        })
        evt = self._wait_for("inited")
        self._attach_rings(evt)

        # kick streaming (if your worker requires explicit start)
        self.req_q.put({"type": "start"})
        self._wait_for("started")

    def _attach_rings(self, evt):
        if self.left_ring:
            self.left_ring.close(); self.left_ring = None
        if self.right_ring:
            self.right_ring.close(); self.right_ring = None
        self.left_ring  = ShmRing(ShmFrameSpec(**evt["left_spec"]),  create=False)
        self.right_ring = ShmRing(ShmFrameSpec(**evt["right_spec"]), create=False)

    def _wait_for(self, typ):
        while True:
            msg = self.evt_q.get()
            if msg.get("type") == typ:
                return msg
            if msg.get("type") == "error":
                raise RuntimeError(f"Dual camera worker error: {msg['msg']}\n{msg.get('tb','')}")

    def _wait_for_any(self, types):
        while True:
            msg = self.evt_q.get()
            t = msg.get("type")
            if t in types:
                return msg
            if t == "error":
                raise RuntimeError(f"Dual camera worker error: {msg['msg']}\n{msg.get('tb','')}")
            # ignore frames and other noise during control ops

    def set_configs(self, cfg_left, cfg_right):
        """
        Send new configs. If shapes change, the worker will emit 'reshaped',
        we reattach to new SM and ack so the worker can resume streaming.
        """
        self.req_q.put({"type": "set_configs", "cfg_left": cfg_left, "cfg_right": cfg_right})

        while True:
            msg = self._wait_for_any({"config_applied", "reshaped"})
            t = msg["type"]
            if t == "config_applied":
                return
            if t == "reshaped":
                # reattach rings to new shapes
                self._attach_rings(msg)
                # tell worker we're ready to resume
                self.req_q.put({"type": "reshape_ack"})
                # loop to wait for final 'config_applied'
                continue

    def get_frames(self):
        """Blocking read of latest frame indices from event queue."""
        while True:
            msg = self.evt_q.get()
            typ = msg.get("type")
            if typ == "frame":
                li = msg["left_idx"]; ri = msg["right_idx"]
                left  = self.left_ring.read_slot(li)
                right = self.right_ring.read_slot(ri)
                return left, right, msg["ts"]
            elif typ == "reshaped":
                # if a consumer thread is reading, reattach here too
                self._attach_rings(msg)
                self.req_q.put({"type": "reshape_ack"})
            elif typ == "error":
                raise RuntimeError(f"Dual camera worker error: {msg['msg']}\n{msg.get('tb','')}")
            # ignore other control events during blocking get

    def shutdown(self):
        if self.req_q:
            self.req_q.put({"type": "shutdown"})
        if self.proc:
            self.proc.join(timeout=2)
        if self.left_ring:
            self.left_ring.close()
        if self.right_ring:
            self.right_ring.close()
