# dual_camera_worker.py
import multiprocessing as mp
import queue
import time, traceback
import numpy as np

from brillouin_system.devices.cameras.allied.allied_config.allied_config import allied_config
from brillouin_system.helpers.frame_ipc_shared import ShmRing, ShmFrameSpec

def dual_camera_worker(req_q: mp.Queue, evt_q: mp.Queue):
    left_ring = right_ring = None
    li = ri = 0
    running = False
    cams = None

    # track naming for shared memory and generation for reshapes
    left_base_name = right_base_name = None
    gen = 0

    try:
        while True:
            try:
                cmd = req_q.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd:
                typ = cmd["type"]

                if typ == "init":
                    use_dummy = cmd.get("use_dummy", False)
                    if use_dummy:
                        from brillouin_system.devices.cameras.allied.dual.dummy_dual_cameras import DummyDualCameras
                        cams = DummyDualCameras()
                    else:
                        from brillouin_system.devices.cameras.allied.dual.dual_allied_vision_cameras import DualAlliedVisionCameras
                        cams = DualAlliedVisionCameras()
                    print(f"[DualCameraWorker] Using {'Dummy' if use_dummy else 'Real'} cameras.")

                    left_cfg  = allied_config["left"].get()
                    right_cfg = allied_config["right"].get()
                    shape_left  = (left_cfg.height,  left_cfg.width)
                    shape_right = (right_cfg.height, right_cfg.width)

                    # keep base names so we can version them if we reshape later
                    left_base_name  = cmd["left_spec"]["name"]
                    right_base_name = cmd["right_spec"]["name"]
                    gen = 0

                    ls = ShmFrameSpec(name=left_base_name,
                                      shape=shape_left,
                                      dtype=cmd["left_spec"]["dtype"],
                                      slots=cmd["left_spec"]["slots"])
                    rs = ShmFrameSpec(name=right_base_name,
                                      shape=shape_right,
                                      dtype=cmd["right_spec"]["dtype"],
                                      slots=cmd["right_spec"]["slots"])
                    left_ring  = ShmRing(ls, create=True)
                    right_ring = ShmRing(rs, create=True)

                    cams.set_configs(left_cfg, right_cfg)

                    evt_q.put({"type": "inited",
                               "left_spec": ls.__dict__,
                               "right_spec": rs.__dict__})

                elif typ == "start":
                    running = True
                    evt_q.put({"type": "started"})

                elif typ == "stop":
                    running = False
                    evt_q.put({"type": "stopped"})

                elif typ == "set_configs":
                    if not cams:
                        evt_q.put({"type": "error", "msg": "Cameras not initialized"})
                    else:
                        # extract shapes implied by new configs; if None, keep existing
                        cfg_left  = cmd.get("cfg_left")
                        cfg_right = cmd.get("cfg_right")

                        # allow ThreadSafeConfig: .get()
                        if hasattr(cfg_left, "get"):  cfg_left  = cfg_left.get()
                        if hasattr(cfg_right, "get"): cfg_right = cfg_right.get()

                        new_left_shape  = (cfg_left.height,  cfg_left.width)   if cfg_left  else left_ring.spec.shape
                        new_right_shape = (cfg_right.height, cfg_right.width)  if cfg_right else right_ring.spec.shape

                        shapes_changed = (tuple(new_left_shape)  != tuple(left_ring.spec.shape) or
                                          tuple(new_right_shape) != tuple(right_ring.spec.shape))

                        # always apply camera configs
                        cams.set_configs(cfg_left, cfg_right)

                        if not shapes_changed:
                            evt_q.put({"type": "config_applied"})
                        else:
                            # pause output and reallocate rings with new names
                            was_running = running
                            running = False

                            # snapshot BEFORE closing (handles both reshape and first-time allocate defensively)
                            old_left_slots = left_ring.spec.slots if left_ring else cmd["left_spec"]["slots"]
                            old_right_slots = right_ring.spec.slots if right_ring else cmd["right_spec"]["slots"]
                            old_left_dtype = left_ring.spec.dtype if left_ring else cmd["left_spec"]["dtype"]
                            old_right_dtype = right_ring.spec.dtype if right_ring else cmd["right_spec"]["dtype"]

                            # close + unlink old rings
                            if left_ring:
                                left_ring.close()
                                left_ring.unlink()
                            if right_ring:
                                right_ring.close()
                                right_ring.unlink()

                            # bump generation and build new names
                            gen += 1
                            new_left_name = f"{left_base_name}_g{gen}"
                            new_right_name = f"{right_base_name}_g{gen}"

                            # allocate new rings with NEW SHAPES, but SAME dtype/slots as before
                            ls = ShmFrameSpec(
                                name=new_left_name,
                                shape=tuple(new_left_shape),  # (H, W)
                                dtype=old_left_dtype,  # keep dtype (e.g., 'uint8')
                                slots=old_left_slots,  # keep ring depth
                            )
                            rs = ShmFrameSpec(
                                name=new_right_name,
                                shape=tuple(new_right_shape),
                                dtype=old_right_dtype,
                                slots=old_right_slots,
                            )

                            left_ring = ShmRing(ls, create=True)
                            right_ring = ShmRing(rs, create=True)
                            li = ri = 0

                            # notify proxy to reattach, then wait for ack
                            evt_q.put({"type": "reshaped",
                                       "left_spec":  ls.__dict__,
                                       "right_spec": rs.__dict__})

                            # wait for proxy to confirm it has reattached before resuming
                            while True:
                                try:
                                    ack = req_q.get(timeout=0.1)
                                except queue.Empty:
                                    ack = None
                                if ack and ack.get("type") == "reshape_ack":
                                    break
                                # allow stop/start while waiting
                                if ack and ack.get("type") == "stop":
                                    was_running = False
                                    evt_q.put({"type": "stopped"})
                                if ack and ack.get("type") == "start":
                                    was_running = True
                                    evt_q.put({"type": "started"})

                            running = was_running
                            evt_q.put({"type": "config_applied"})

                elif typ == "shutdown":
                    break

            if running and cams and left_ring and right_ring:
                f0, f1 = cams.snap_once(timeout=5.0)

                # Normalize to np.uint8
                if hasattr(f0, "as_numpy_ndarray"):
                    left_frame  = f0.as_numpy_ndarray().astype(np.uint8)
                    right_frame = f1.as_numpy_ndarray().astype(np.uint8)
                else:
                    left_frame, right_frame = f0, f1

                left_ring.write_slot(li, left_frame)
                right_ring.write_slot(ri, right_frame)
                evt_q.put({"type": "frame",
                           "left_idx": li, "right_idx": ri,
                           "ts": time.time()})

                li = (li + 1) % left_ring.spec.slots
                ri = (ri + 1) % right_ring.spec.slots

    except Exception as e:
        evt_q.put({"type": "error", "msg": str(e), "tb": traceback.format_exc()})
    finally:
        if cams:
            cams.close()
        if left_ring:
            left_ring.close(); left_ring.unlink()
        if right_ring:
            right_ring.close(); right_ring.unlink()
