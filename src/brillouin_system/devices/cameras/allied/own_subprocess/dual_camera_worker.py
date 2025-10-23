# dual_camera_worker.py
import multiprocessing as mp
import queue
import time, traceback


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

    def _shape_from_roi(side_obj, fallback_cfg):
        """Return (H, W) from camera.get_roi() if available; else from cfg."""
        try:
            roi = side_obj.get_roi()  # expects dict with Height/Width
            h = int(roi["Height"])
            w = int(roi["Width"])
            return (h, w)
        except Exception:
            return (int(fallback_cfg.height), int(fallback_cfg.width))

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

                    # 1) Apply current configs to hardware
                    left_cfg  = allied_config["left"].get()
                    right_cfg = allied_config["right"].get()
                    cams.set_configs(left_cfg, right_cfg)

                    # 2) Query actual accepted shapes
                    shape_left  = _shape_from_roi(cams.left,  left_cfg)
                    shape_right = _shape_from_roi(cams.right, right_cfg)

                    # 3) Allocate rings from actual shapes
                    left_base_name  = cmd["left_spec"]["name"]
                    right_base_name = cmd["right_spec"]["name"]
                    gen = 0

                    ls = ShmFrameSpec(
                        name=left_base_name,
                        shape=shape_left,
                        dtype=cmd["left_spec"]["dtype"],
                        slots=cmd["left_spec"]["slots"],
                    )
                    rs = ShmFrameSpec(
                        name=right_base_name,
                        shape=shape_right,
                        dtype=cmd["right_spec"]["dtype"],
                        slots=cmd["right_spec"]["slots"],
                    )
                    left_ring  = ShmRing(ls, create=True)
                    right_ring = ShmRing(rs, create=True)

                    evt_q.put({
                        "type": "inited",
                        "left_spec": ls.__dict__,
                        "right_spec": rs.__dict__,
                    })

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
                        cfg_left  = cmd.get("cfg_left")
                        cfg_right = cmd.get("cfg_right")
                        if hasattr(cfg_left, "get"):  cfg_left  = cfg_left.get()
                        if hasattr(cfg_right, "get"): cfg_right = cfg_right.get()

                        # 1) Apply to hardware
                        cams.set_configs(cfg_left, cfg_right)

                        # 2) Query actual accepted shapes
                        #    (Use latest known configs as fallback if get_roi is unavailable)
                        left_cfg_fallback  = cfg_left  if cfg_left  is not None else allied_config["left"].get()
                        right_cfg_fallback = cfg_right if cfg_right is not None else allied_config["right"].get()

                        new_left_shape  = _shape_from_roi(cams.left,  left_cfg_fallback)
                        new_right_shape = _shape_from_roi(cams.right, right_cfg_fallback)

                        # 3) Reshape only if actual shapes changed
                        if (tuple(new_left_shape)  == tuple(left_ring.spec.shape) and
                            tuple(new_right_shape) == tuple(right_ring.spec.shape)):
                            evt_q.put({"type": "config_applied"})
                        else:
                            was_running = running
                            running = False

                            # snapshot old meta
                            old_left_slots  = left_ring.spec.slots
                            old_right_slots = right_ring.spec.slots
                            old_left_dtype  = left_ring.spec.dtype
                            old_right_dtype = right_ring.spec.dtype

                            # close + unlink old rings
                            left_ring.close();  left_ring.unlink()
                            right_ring.close(); right_ring.unlink()

                            # bump generation and build new names
                            gen += 1
                            new_left_name  = f"{left_base_name}_g{gen}"
                            new_right_name = f"{right_base_name}_g{gen}"

                            # allocate with ACTUAL shapes
                            ls = ShmFrameSpec(
                                name=new_left_name,
                                shape=new_left_shape,
                                dtype=old_left_dtype,
                                slots=old_left_slots,
                            )
                            rs = ShmFrameSpec(
                                name=new_right_name,
                                shape=new_right_shape,
                                dtype=old_right_dtype,
                                slots=old_right_slots,
                            )

                            left_ring  = ShmRing(ls, create=True)
                            right_ring = ShmRing(rs, create=True)
                            li = ri = 0

                            # notify proxy and wait for ACK
                            evt_q.put({
                                "type": "reshaped",
                                "left_spec":  ls.__dict__,
                                "right_spec": rs.__dict__,
                            })
                            while True:
                                try:
                                    ack = req_q.get(timeout=0.1)
                                except queue.Empty:
                                    ack = None
                                if ack and ack.get("type") == "reshape_ack":
                                    break
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

            # ---- streaming loop ----
            if running and cams and left_ring and right_ring:
                f0, f1 = cams.snap_once(timeout=5.0)

                # accept frames as returned (no dtype/shape normalization here)
                if hasattr(f0, "as_numpy_ndarray"):
                    left_frame  = f0.as_numpy_ndarray()
                    right_frame = f1.as_numpy_ndarray()
                else:
                    left_frame, right_frame = f0, f1

                # Guard against mismatches (avoid assert in write_slot)
                if (tuple(getattr(left_frame,  "shape", ()))  != tuple(left_ring.spec.shape) or
                    tuple(getattr(right_frame, "shape", ())) != tuple(right_ring.spec.shape)):
                    # Optionally emit a diagnostic; skip this pair
                    evt_q.put({
                        "type": "warn",
                        "msg": f"Frame/Spec mismatch: L {getattr(left_frame,'shape',None)} vs {left_ring.spec.shape}, "
                               f"R {getattr(right_frame,'shape',None)} vs {right_ring.spec.shape}"
                    })
                    continue

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
