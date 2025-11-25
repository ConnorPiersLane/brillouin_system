# dual_camera_worker.py
import multiprocessing as mp
import queue
import time, traceback


from brillouin_system.eye_tracker.eye_tracker import IMG_SIZE, EyeTrackerResultsForGui
from brillouin_system.helpers.frame_ipc_shared import ShmRing, ShmFrameSpec

SLOTS = 2
DTYPE = "uint8"  # Must comply with eyetracker

def eye_tracker_worker(req_q: mp.Queue, evt_q: mp.Queue):

    img_shape = (IMG_SIZE[0], IMG_SIZE[1], 3) # All images are made to rgb

    left_ring = right_ring = rendered_ring = xymap_ring = None
    left_i = right_i = rendered_i = xymap_i = 0
    running = False
    eye_tracker = None

    try:
        while True:
            try:
                cmd = req_q.get_nowait()
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
                    process_name = cmd['name'] # TODO: I left here

                    left_base_name  = cmd["left_spec"]["name"]
                    right_base_name = cmd["right_spec"]["name"]
                    rendered_base_name  = cmd["rendered_spec"]["name"]
                    xymap_base_name = cmd["xymap_spec"]["name"]

                    left_s = ShmFrameSpec(
                        name=left_base_name,
                        shape=img_shape,
                        dtype=cmd["left_spec"]["dtype"],
                        slots=cmd["left_spec"]["slots"],
                    )
                    right_s = ShmFrameSpec(
                        name=right_base_name,
                        shape=img_shape,
                        dtype=cmd["right_spec"]["dtype"],
                        slots=cmd["right_spec"]["slots"],
                    )
                    rendered_s = ShmFrameSpec(
                        name=rendered_base_name,
                        shape=img_shape,
                        dtype=cmd["rendered_spec"]["dtype"],
                        slots=cmd["rendered_spec"]["slots"],
                    )
                    xymap_s = ShmFrameSpec(
                        name=xymap_base_name,
                        shape=img_shape,
                        dtype=cmd["xymap_spec"]["dtype"],
                        slots=cmd["xymap_spec"]["slots"],
                    )
                    left_ring  = ShmRing(left_s, create=True)
                    right_ring = ShmRing(right_s, create=True)
                    rendered_ring = ShmRing(rendered_s, create=True)
                    xymap_ring = ShmRing(xymap_s, create=True)

                    evt_q.put({
                        "type": "inited",
                        "left_spec": left_s.__dict__,
                        "right_spec": right_s.__dict__,
                        "rendered_spec": rendered_s.__dict__,
                        "xymap_spec": xymap_s.__dict__,
                    })

                elif typ == "start":
                    running = True
                    evt_q.put({"type": "started"})

                elif typ == "stop":
                    running = False
                    evt_q.put({"type": "stopped"})

                elif typ == "set_allied_configs":
                    cfg_left  = cmd.get("cfg_left")
                    cfg_right = cmd.get("cfg_right")

                    # 1) Apply to hardware
                    eye_tracker.set_allied_vision_configs(cfg_left, cfg_right)
                    evt_q.put({"type": "allied_configs_applied"})

                elif typ == "set_et_config":
                    cfg_et = cmd.get("cfg_et")
                    eye_tracker.set_config(cfg_et)
                    evt_q.put({"type": "et_config_applied"})

                elif typ == "start_saving":
                    eye_tracker.start_saving()
                    evt_q.put({"type": "started_saving"})

                elif typ == "end_saving":
                    eye_tracker.start_saving()
                    evt_q.put({"type": "ended_saving"})

                elif typ == "shutdown":
                    break

            # ---- streaming loop ----
            if running and eye_tracker and left_ring and right_ring and rendered_ring and xymap_ring:
                results_for_gui: EyeTrackerResultsForGui = eye_tracker.get_results_for_gui()
                left_frame = results_for_gui.cam_left_img
                right_frame = results_for_gui.cam_right_img
                rendered_frame = results_for_gui.rendered_img
                xymap_frame = results_for_gui.xymap_img


                # Guard against mismatches (avoid assert in write_slot)
                if (tuple(getattr(left_frame,  "shape", ()))  != tuple(left_ring.spec.shape) or
                    tuple(getattr(right_frame, "shape", ())) != tuple(right_ring.spec.shape) or
                    tuple(getattr(rendered_frame, "shape", ())) != tuple(rendered_ring.spec.shape) or
                    tuple(getattr(xymap_frame, "shape", ())) != tuple(xymap_ring.spec.shape)):
                    # Optionally emit a diagnostic; skip this pair
                    evt_q.put({
                        "type": "warn",
                        "msg": f"Frame/Spec mismatch: L {getattr(left_frame,'shape',None)} vs {left_ring.spec.shape}, "
                               f"R {getattr(right_frame,'shape',None)} vs {right_ring.spec.shape}, "
                               f"R {getattr(rendered_frame, 'shape', None)} vs {rendered_ring.spec.shape}, "
                               f"R {getattr(xymap_frame, 'shape', None)} vs {xymap_ring.spec.shape}, "
                    })
                    continue

                left_ring.write_slot(left_i, left_frame)
                right_ring.write_slot(right_i, right_frame)
                rendered_ring.write_slot(rendered_i, rendered_frame)
                xymap_ring.write_slot(xymap_i, xymap_frame)

                evt_q.put({"type": "frame",
                           "left_idx": left_i, "right_idx": right_i, "rendered_idx": rendered_i, "xymap_idx": xymap_i,
                           "ts": time.time()})
                left_i = (left_i + 1) % left_ring.spec.slots
                right_i = (right_i + 1) % right_ring.spec.slots
                rendered_i = (rendered_i + 1) % rendered_ring.spec.slots
                xymap_i = (xymap_i + 1) % xymap_ring.spec.slots

    except Exception as e:
        evt_q.put({"type": "error", "msg": str(e), "tb": traceback.format_exc()})
    finally:
        if eye_tracker:
            eye_tracker.shutdown()

        if left_ring:
            left_ring.close(); left_ring.unlink()
        if right_ring:
            right_ring.close(); right_ring.unlink()
        if rendered_ring:
            rendered_ring.close(); rendered_ring.unlink()
        if xymap_ring:
            xymap_ring.close(); xymap_ring.unlink()
