import time

import numpy as np

from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra



def test_ixon_camera():
    print("=== Starting IxonUltra camera test (no cooling) ===")

    cam = IxonUltra(index=0, temperature=10, fan_mode="full")

    cam.set_fixed_pre_amp_mode(0)
    cam.set_vss_index(0)
    cam.set_flip_image_horizontally(False)
    cam.set_exposure_time(1e-3)

    cam.set_roi(2, 73, 5, 30,)
    cam.set_binning(1,1)
    cam.start_streaming(buffer_size=200)

    # warm up: get one frame so buffering is “live”
    while cam.get_latest_frame_poll() is None:
        pass
    n = 10
    t = np.empty(n, dtype=np.float64)
    shapes = []

    for i in range(n):
        # wait until we actually get a frame (non-blocking poll)
        frame = None
        while frame is None:
            frame = cam.get_latest_frame_poll()

        t[i] = time.perf_counter()
        if i == 0:
            shapes.append(frame.shape)

    cam.stop_streaming()

    dt = (t[1:] - t[:-1]) * 1000.0
    print("shape:", shapes[0])
    print("Δt ms:", dt)
    print(f"mean Δt: {dt.mean():.3f} ms, median Δt: {np.median(dt):.3f} ms, fps~{1000.0/dt.mean():.1f}")


    print("=== Test complete ===")

    def dump_methods(obj, keywords):
        names = sorted(dir(obj))
        for kw in keywords:
            hits = [n for n in names if kw in n.lower()]
            if hits:
                print(f"\n--- contains '{kw}' ---")
                for h in hits:
                    print(" ", h)

    # dump_methods(cam.cam, ["frame", "transfer", "overlap", "kinetic", "cycle", "timing", "trigger", "read", "acq"])
    print("read_mode:", cam.cam.get_read_mode())
    print("readmode_desc:", getattr(cam.cam, "_readmode_desc", None))
    print("kinetic params:", cam.cam.get_kinetic_mode_parameters())
    print("fast kinetic params:", cam.cam.get_fast_kinetic_mode_parameters())
    print("frame_timings:", cam.cam.get_frame_timings())
    print("cycle_timings:", cam.cam.get_cycle_timings())
    print("readout_time:", cam.cam.get_readout_time())

    print("kinetic params:", cam.cam.get_kinetic_mode_parameters())
    print("fast kinetic params:", cam.cam.get_fast_kinetic_mode_parameters())


    cam.close()


if __name__ == "__main__":
    test_ixon_camera()
