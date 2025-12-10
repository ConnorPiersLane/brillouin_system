
# eye_tracker_proxy.py
import multiprocessing as mp
import queue
from typing import Tuple, Optional, Dict

import numpy as np

from brillouin_system.helpers.frame_ipc_shared import ShmFrameSpec, ShmRing


class EyeTrackerProxy:
    """
    Proxy for the eye_tracker_worker subprocess.

    Responsibilities
    ----------------
    - Spawn the worker in a separate process.
    - Send control commands via req_q ("init", "start", "stop", "set_*", "shutdown").
    - Attach to the shared-memory rings created by the worker.
    - Provide convenient APIs to fetch frames (left, right).
    """

    def __init__(self, use_dummy: bool = False):
        self.req_q: mp.Queue = mp.Queue()
        self.evt_q: mp.Queue = mp.Queue(maxsize=16)
        self.proc: Optional[mp.Process] = None

        self.left_ring: Optional[ShmRing] = None
        self.right_ring: Optional[ShmRing] = None

        self.use_dummy = use_dummy

        # base name used by the worker to construct shared memory names
        self._base_name = f"eye_{mp.current_process().name}"

    def _make_process(self) -> mp.Process:
        # NOTE: adjust this import path to wherever your worker actually lives.
        from brillouin_system.eye_tracker.eye_tracker_worker import eye_tracker_worker
        return mp.Process(target=eye_tracker_worker, args=(self.req_q, self.evt_q), daemon=False)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def start(self) -> None:
        """
        Start the worker process, initialize shared memory rings, and begin streaming.
        """
        if self.proc is None or not self.proc.is_alive():
            self.proc = self._make_process()
            self.proc.start()

        # Tell worker to init and create rings.
        self.req_q.put({
            "type": "init",
            "name": self._base_name,
            "use_dummy": self.use_dummy,
        })

        evt = self._wait_for("inited")
        self._attach_rings(evt)

        # Kick streaming.
        self.req_q.put({"type": "start"})
        self._wait_for("started")

    def shutdown(self) -> None:
        """
        Ask the worker to shut down and clean up local resources.
        """
        try:
            if self.req_q:
                self.req_q.put({"type": "shutdown"})
        except Exception:
            pass

        if self.proc:
            self.proc.join(timeout=2)

        if self.left_ring:
            self.left_ring.close()
            self.left_ring = None
        if self.right_ring:
            self.right_ring.close()
            self.right_ring = None


    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _attach_rings(self, evt: dict) -> None:
        """
        Attach ShmRing instances to the specs provided by the worker's 'inited' event.
        """
        if self.left_ring:
            self.left_ring.close()
        if self.right_ring:
            self.right_ring.close()

        self.left_ring = ShmRing(ShmFrameSpec(**evt["left_spec"]), create=False)
        self.right_ring = ShmRing(ShmFrameSpec(**evt["right_spec"]), create=False)


    def _wait_for(self, typ: str) -> dict:
        """
        Block until an event of the given type is seen or an error is reported.
        """
        while True:
            msg = self.evt_q.get()
            if msg.get("type") == typ:
                return msg
            if msg.get("type") == "error":
                raise RuntimeError(
                    f"Eye tracker worker error: {msg['msg']}\n{msg.get('tb', '')}"
                )

    def _wait_for_any(self, types) -> dict:
        """
        Block until an event whose type is in `types` is seen, or an error is reported.
        """
        types = set(types)
        while True:
            msg = self.evt_q.get()
            t = msg.get("type")
            if t in types:
                return msg
            if t == "error":
                raise RuntimeError(
                    f"Eye tracker worker error: {msg['msg']}\n{msg.get('tb', '')}"
                )
            # Ignore unrelated events (e.g. "frame", "warn") for control operations.

    # -------------------------------------------------------------------------
    # Config & control APIs
    # -------------------------------------------------------------------------
    def set_allied_configs(self, cfg_left, cfg_right) -> None:
        """
        Push new Allied Vision camera configs into the EyeTracker.
        If the worker reshapes its rings, we reattach and ACK the reshape.
        """
        self.req_q.put({
            "type": "set_allied_configs",
            "cfg_left": cfg_left,
            "cfg_right": cfg_right,
        })
        self._wait_for("allied_configs_applied")

    def set_et_config(self, cfg_et) -> None:
        """
        Push new eye tracker configuration.
        """
        self.req_q.put({"type": "set_et_config", "cfg_et": cfg_et})
        self._wait_for("et_config_applied")


    def stop_streaming(self) -> None:
        """
        Stop the streaming loop in the worker (but keep process alive).
        """
        self.req_q.put({"type": "stop"})
        self._wait_for("stopped")

    def start_streaming(self) -> None:
        """
        Start (or resume) the streaming loop in the worker.
        """
        self.req_q.put({"type": "start"})
        self._wait_for("started")

    # -------------------------------------------------------------------------
    # Frame access
    # -------------------------------------------------------------------------
    def get_frame(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Block until the next frame event arrives, then return the three images
        (left, right) and metadata.

        Returns
        -------
        (left, right, meta)
          left     : np.ndarray (H, W, 3), uint8
          right    : np.ndarray (H, W, 3), uint8
          meta     : Dict: {"ts": last["ts"], "idx": last["idx"], "pupil3D": pupil3D}
        """
        while True:
            msg = self.evt_q.get()
            typ = msg.get("type")

            if typ == "frame":
                idx = msg["idx"]
                if not (self.left_ring and self.right_ring):
                    raise RuntimeError("Rings not attached")

                left = self.left_ring.read_slot(idx)
                right = self.right_ring.read_slot(idx)

                p3d_dict = msg.get("pupil3D")
                if p3d_dict is not None:
                    from brillouin_system.eye_tracker.pupil_fitting.pupil3D import Pupil3D
                    pupil3D = Pupil3D(
                        center_left=np.array(p3d_dict["center_left"]) if p3d_dict["center_left"] is not None else None,
                        center_ref=np.array(p3d_dict["center_ref"]) if p3d_dict["center_ref"] is not None else None,
                        normal_left=np.array(p3d_dict["normal_left"]) if p3d_dict["normal_left"] is not None else None,
                        normal_ref=np.array(p3d_dict["normal_ref"]) if p3d_dict["normal_ref"] is not None else None,
                        radius=p3d_dict["radius"],
                    )
                else:
                    pupil3D = None

                meta = {"ts": msg["ts"], "idx": msg["idx"], "pupil3D": pupil3D}

                return left, right, meta

            elif typ in ("warn", "started", "stopped", "inited",
                         "allied_configs_applied", "et_config_applied",
                         "started_saving", "ended_saving"):
                # Ignore non-frame events.
                continue

            elif typ == "error":
                raise RuntimeError(
                    f"Eye tracker worker error: {msg['msg']}\n{msg.get('tb', '')}"
                )

    def get_latest(
        self, timeout: Optional[float] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Return the latest frame triple currently queued, or block up to `timeout`
        to wait for one.

        If no usable frame arrives, returns None.
        (left, right, meta)
          left     : np.ndarray (H, W, 3), uint8
          right    : np.ndarray (H, W, 3), uint8
          meta     : Dict: {"ts": last["ts"], "idx": last["idx"], "pupil3D": pupil3D}
        """
        last = None

        # Drain queue non-blocking first.
        while True:
            try:
                msg = self.evt_q.get_nowait()
            except queue.Empty:
                break

            t = msg.get("type")
            if t == "frame":
                last = msg
            elif t == "error":
                raise RuntimeError(
                    f"Eye tracker worker error: {msg['msg']}\n{msg.get('tb', '')}"
                )
            # ignore warn/control messages

        # If nothing in backlog, optionally block for one.
        if last is None and timeout is not None:
            try:
                msg = self.evt_q.get(timeout=timeout)
            except queue.Empty:
                return None

            t = msg.get("type")
            if t == "frame":
                last = msg
            elif t == "error":
                raise RuntimeError(
                    f"Eye tracker worker error: {msg['msg']}\n{msg.get('tb', '')}"
                )
            else:
                # Not a frame; treat as no data
                return None

        if last is None:
            return None

        idx = last["idx"]
        if not (self.left_ring and self.right_ring):
            raise RuntimeError("Rings not attached")

        left = self.left_ring.read_slot(idx)
        right = self.right_ring.read_slot(idx)

        # Rebuild Pupil3D from the last frame message
        p3d_dict = last.get("pupil3D")
        if p3d_dict is not None:
            from brillouin_system.eye_tracker.pupil_fitting.pupil3D import Pupil3D
            pupil3D = Pupil3D(
                center_left=np.array(p3d_dict["center_left"]) if p3d_dict["center_left"] is not None else None,
                center_ref=np.array(p3d_dict["center_ref"]) if p3d_dict["center_ref"] is not None else None,
                normal_left=np.array(p3d_dict["normal_left"]) if p3d_dict["normal_left"] is not None else None,
                normal_ref=np.array(p3d_dict["normal_ref"]) if p3d_dict["normal_ref"] is not None else None,
                radius=p3d_dict["radius"],
            )
        else:
            pupil3D = None

        meta = {"ts": last["ts"], "idx": last["idx"], "pupil3D": pupil3D}

        return left, right, meta
