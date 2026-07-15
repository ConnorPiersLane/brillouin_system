# dual_camera_worker.py
import multiprocessing as mp
import queue
import time, traceback


from brillouin_system.eye_tracker.eye_tracker import IMG_SIZE, EyeTrackerResultsForGui
from brillouin_system.helpers.frame_ipc_shared import ShmRing, ShmFrameSpec

SLOTS = 16
DTYPE = "uint8"  # Must comply with eyetracker


def _put_evt(evt_q, evt):
    """Best-effort put that never blocks; drops the oldest event if the queue is full."""
    try:
        evt_q.put_nowait(evt)
    except queue.Full:
        try:
            evt_q.get_nowait()
        except queue.Empty:
            pass
        try:
            evt_q.put_nowait(evt)
        except queue.Full:
            pass


def eye_tracker_worker(req_q: mp.Queue, evt_q: mp.Queue):

    img_shape = (IMG_SIZE[0], IMG_SIZE[1], 3) # All images are made to rgb

    left_ring = right_ring = None
    idx = 0
    running = False
    eye_tracker = None

    try:
        while True:
            try:
                if running:
                    cmd = req_q.get_nowait()
                else:
                    # Idle: block briefly instead of busy-spinning at 100% CPU.
                    cmd = req_q.get(timeout=0.05)
            except queue.Empty:
                cmd = None

            if cmd:
                typ = cmd["type"]

                if typ == "init":
                    from brillouin_system.eye_tracker.eye_tracker import EyeTracker
                    use_dummy = cmd.get("use_dummy", False)
                    eye_tracker = EyeTracker(use_dummy=use_dummy)

                    print(f"[eye_tracker_worker] Using {'Dummy' if use_dummy else 'Real'} cameras.")


                    # 1) Allocate rings
                    process_name = cmd['name']

                    left_base_name  = f"{process_name}_left"
                    right_base_name = f"{process_name}_right"

                    left_s = ShmFrameSpec(
                        name=left_base_name,
                        shape=img_shape,
                        dtype=DTYPE,
                        slots=SLOTS,
                    )
                    right_s = ShmFrameSpec(
                        name=right_base_name,
                        shape=img_shape,
                        dtype=DTYPE,
                        slots=SLOTS,
                    )

                    left_ring  = ShmRing(left_s, create=True)
                    right_ring = ShmRing(right_s, create=True)

                    evt_q.put({
                        "type": "inited",
                        "left_spec": left_s.__dict__,
                        "right_spec": right_s.__dict__,
                    })

                elif typ == "start":
                    running = True
                    evt_q.put({"type": "started"})

                elif typ == "stop":
                    running = False
                    evt_q.put({"type": "stopped"})

                elif typ == "set_allied_configs":
                    if not eye_tracker:
                        evt_q.put({"type": "error", "msg": "EyeTracker not initialized"})
                    else:
                        cfg_left  = cmd.get("cfg_left")
                        cfg_right = cmd.get("cfg_right")

                        # 1) Apply to hardware
                        eye_tracker.set_allied_vision_configs(cfg_left, cfg_right)
                        evt_q.put({"type": "allied_configs_applied"})

                elif typ == "set_et_config":
                    if not eye_tracker:
                        evt_q.put({"type": "error", "msg": "EyeTracker not initialized"})
                    else:
                        cfg_et = cmd.get("cfg_et")
                        eye_tracker.set_config(cfg_et)
                        evt_q.put({"type": "et_config_applied"})

                elif typ == "shutdown":
                    break

            # ---- streaming loop ----
            if running and eye_tracker and left_ring and right_ring:
                try:
                    results_for_gui: EyeTrackerResultsForGui = eye_tracker.get_results_for_gui()
                    left_frame = results_for_gui.cam_left_img
                    right_frame = results_for_gui.cam_right_img

                    p3d = results_for_gui.pupil3D
                    if p3d is not None:
                        p3d_dict = {
                            "center_left": p3d.center_left.tolist() if p3d.center_left is not None else None,
                            "center_ref": p3d.center_ref.tolist() if p3d.center_ref is not None else None,
                            "normal_left": p3d.normal_left.tolist() if p3d.normal_left is not None else None,
                            "normal_ref": p3d.normal_ref.tolist() if p3d.normal_ref is not None else None,
                            "radius": p3d.radius,
                        }
                    else:
                        p3d_dict = None

                    # Guard against mismatches (avoid assert in write_slot)
                    if (tuple(getattr(left_frame,  "shape", ()))  != tuple(left_ring.spec.shape) or
                        tuple(getattr(right_frame, "shape", ())) != tuple(right_ring.spec.shape)
                        ):
                        # Optionally emit a diagnostic; skip this pair
                        _put_evt(evt_q, {
                            "type": "warn",
                            "msg": f"Frame/Spec mismatch: Left {getattr(left_frame,'shape',None)} vs {left_ring.spec.shape}, "
                                   f"Right {getattr(right_frame,'shape',None)} vs {right_ring.spec.shape}, "
                        })
                        continue

                    left_ring.write_slot(idx, left_frame)
                    right_ring.write_slot(idx, right_frame)

                    evt_q.put({
                        "type": "frame",
                        "idx": idx,
                        "ts": time.time(),
                        "pupil3D": p3d_dict,
                    })
                    idx = (idx + 1) % SLOTS

                except Exception as e:
                    # Camera (or tracker) crashed mid-stream. Stop streaming and
                    # report upstream, but KEEP this process alive so the GUI's
                    # shutdown/ReStart handshake still works.
                    running = False
                    tb = traceback.format_exc()
                    print("[eye_tracker_worker] CAMERA/TRACKER ERROR — streaming stopped:")
                    print(tb, flush=True)
                    _put_evt(evt_q, {"type": "error", "msg": str(e), "tb": tb})

    except Exception as e:
        evt_q.put({"type": "error", "msg": str(e), "tb": traceback.format_exc()})
    finally:
        if eye_tracker:
            eye_tracker.shutdown()

        if left_ring:
            left_ring.close(); left_ring.unlink()
        if right_ring:
            right_ring.close(); right_ring.unlink()



