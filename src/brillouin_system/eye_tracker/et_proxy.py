import multiprocessing as mp

class EyeTrackerProxy:
    def __init__(self, proxy_kwargs):
        self.req_q = mp.Queue()
        self.evt_q = mp.Queue(maxsize=4)
        self.proc = None
        self.proxy_kwargs = proxy_kwargs

    def start(self):
        if self.proc is None or not self.proc.is_alive():
            from et_worker import running_eye_tracker_worker
            self.proc = mp.Process(target=running_eye_tracker_worker,
                                   args=(self.req_q, self.evt_q, self.proxy_kwargs),
                                   daemon=True)
            self.proc.start()

        # Wait for worker to signal it has started
        self._wait_for("started")

    def get_result(self):
        """Blocking read of latest tracking result."""
        while True:
            msg = self.evt_q.get()
            if msg.get("type") == "result":
                return msg["data"]
            elif msg.get("type") == "error":
                raise RuntimeError(f"EyeTracker worker error: {msg['msg']}\n{msg.get('tb','')}")
            # ignore other messages

    def set_config(self, cfg):
        """Send configuration update to worker."""
        self.req_q.put({"type": "config", "data": cfg})

    def shutdown(self):
        self.req_q.put({"type": "shutdown"})
        if self.proc:
            self.proc.join(timeout=2)

    def _wait_for(self, typ):
        while True:
            msg = self.evt_q.get()
            if msg.get("type") == typ:
                return msg
            if msg.get("type") == "error":
                raise RuntimeError(f"EyeTracker worker error: {msg['msg']}\n{msg.get('tb','')}")
