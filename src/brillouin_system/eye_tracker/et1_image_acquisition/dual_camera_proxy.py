# dual_camera_proxy.py
import multiprocessing as mp

import numpy as np

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
        from brillouin_system.eye_tracker.et1_image_acquisition.dual_camera_worker import dual_camera_worker
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

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Block until the next paired frame event arrives, then return the images and metadata.

        Behavior
        --------
        - Reads a single message from the worker's event queue (evt_q).
        - When it sees a "frame" event, it pulls the two indices (left_idx/right_idx),
          reads those slots from the shared-memory rings, makes *copies* (so the caller
          is not affected by subsequent overwrites), and returns them with metadata.
        - If it sees a "reshaped" event (the worker reallocated rings due to a config
          change that altered width/height), it reattaches to the new rings, ACKs the
          worker, and continues waiting for the first post-reshape frame.
        - Control events like "started"/"stopped"/"inited"/"config_applied" are ignored.

        Shared-memory note
        ------------------
        - The ring readers return arrays that reference shared memory. We call .copy()
          before returning to detach them, so later writes by the worker won't mutate
          what you received. If you *want* zero-copy views instead, remove the .copy()
          calls (but then you must not hold the arrays long).

        Returns
        -------
        (left, right, meta)
            left : np.ndarray
                Left image as uint8, detached from shared memory.
            right : np.ndarray
                Right image as uint8, detached from shared memory.
            meta : dict
                Small metadata dict. Keys may include:
                  - "ts": float wallclock timestamp (worker time.time())
                  - "seq": int monotonically increasing frame sequence (if worker provides)
                  - "tleft"/"tright": hardware timestamps if provided by the worker.

        Raises
        ------
        RuntimeError
            If the worker reports an "error" event.

        Notes
        -----
        - This returns the *next* announced pair, not necessarily the “latest” in the
          rings. With your coalescing worker + small evt_q, backlog is tiny, so this
          is effectively latest under load.
        """
        import queue as _q  # local alias to avoid shadowing

        while True:
            msg = self.evt_q.get()
            typ = msg.get("type")

            if typ == "frame":
                li = msg["left_idx"];
                ri = msg["right_idx"]
                left = self.left_ring.read_slot(li).copy()  # detach from shared memory
                right = self.right_ring.read_slot(ri).copy()  # detach from shared memory
                meta = {}
                for k in ("ts", "seq", "tleft", "tright"):
                    if k in msg:
                        meta[k] = msg[k]
                return left, right, meta

            elif typ == "reshaped":
                # Reattach to new rings and acknowledge so worker can resume.
                self._attach_rings(msg)
                self.req_q.put({"type": "reshape_ack"})
                # Optional: if you want to guarantee the *first* frame you return
                # is post-reshape, just continue and wait for the next "frame".
                continue

            elif typ in ("started", "stopped", "inited", "config_applied"):
                # Control events: nothing to return; keep waiting.
                continue

            elif typ == "error":
                raise RuntimeError(f"Dual camera worker error: {msg['msg']}\n{msg.get('tb', '')}")

    def shutdown(self):
        if self.req_q:
            self.req_q.put({"type": "shutdown"})
        if self.proc:
            self.proc.join(timeout=2)
        if self.left_ring:
            self.left_ring.close()
        if self.right_ring:
            self.right_ring.close()
