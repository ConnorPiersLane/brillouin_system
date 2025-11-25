# et_worker.py

import multiprocessing as mp
import queue
import traceback


def running_eye_tracker_worker(req_q: mp.Queue, evt_q: mp.Queue, proxy_kwargs: dict):
    """
    Worker entrypoint run in a separate process.

    - Instantiates EyeTracker (with optional `use_dummy` and initial config from proxy_kwargs)
    - Sends {"type": "started"} once it's ready
    - In a loop:
        * handles control messages from req_q ("shutdown", "config")
        * calls tracker.get_display_frames()
        * pushes {"type": "result", "data": results} into evt_q
          (dropping stale results if the queue is full)
    - On error, sends {"type": "error", "msg": ..., "tb": ...}
    """

    # Import inside the worker process so it plays nicely with mp spawn/fork
    from eye_tracker import EyeTracker  # and EyeTrackerResultsForGui implicitly

    tracker = None

    try:
        # --- 1. Create EyeTracker instance -----------------------------------
        use_dummy = bool(proxy_kwargs.get("use_dummy", False))
        tracker = EyeTracker(use_dummy=use_dummy)

        # Optional: initial EyeTrackerConfig can be passed in proxy_kwargs["config"]
        initial_cfg = proxy_kwargs.get("config")
        if initial_cfg is not None:
            tracker.set_config(initial_cfg)

        # Signal to the parent that we are up and running
        evt_q.put({"type": "started"})

        running = True

        # --- 2. Main worker loop --------------------------------------------
        while running:
            # 2a) Handle control messages (non-blocking)
            try:
                while True:
                    msg = req_q.get_nowait()
                    mtype = msg.get("type")

                    if mtype == "shutdown":
                        running = False
                        break

                    elif mtype == "config":
                        # Expecting an EyeTrackerConfig instance in msg["data"]
                        cfg = msg.get("data")
                        if cfg is not None:
                            tracker.set_config(cfg)

                    # You can add more command types here if needed

            except queue.Empty:
                pass

            if not running:
                break

            # 2b) Run one eye-tracking step
            # This internally blocks on the cameras as needed.
            results = tracker.get_results_for_gui()  # EyeTrackerResultsForGui

            # 2c) Send latest results to the parent
            # We want "latest" semantics: if evt_q is full, drop the oldest.
            try:
                evt_q.put({"type": "result", "data": results}, timeout=0.01)
            except queue.Full:
                # Drop one old result and try once more
                try:
                    evt_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    evt_q.put({"type": "result", "data": results}, timeout=0.01)
                except queue.Full:
                    # If it's still full, just drop this frame; main process is too slow.
                    pass

        # --- 3. Clean shutdown -----------------------------------------------
        if tracker is not None:
            try:
                tracker.shutdown()
            except Exception:
                # We don't want shutdown failures to overwrite any earlier error message
                pass

    except Exception as e:
        # Report error back to parent
        evt_q.put({
            "type": "error",
            "msg": str(e),
            "tb": traceback.format_exc(),
        })

        # Try to shut down tracker even on error
        if tracker is not None:
            try:
                tracker.shutdown()
            except Exception:
                pass
