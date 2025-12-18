from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
import time

def test_ixon_camera():
    print("=== Starting IxonUltra camera test (no cooling) ===")

    cam = IxonUltra(index=0, temperature="off", fan_mode="full")
    cam.set_fixed_pre_amp_mode(0)
    cam.set_vss_index(0)
    cam.set_flip_image_horizontally(False)
    cam.set_exposure_time(1e-3)
    print(cam.get_exposure_time())
    cam.set_roi(0, 128, 0, 512,)
    cam.set_binning(4,25)
    cam.cam.set_read_mode('fvb')
    # 3) snap one frame
    cam.snap();
    t0 = time.perf_counter();
    cam.snap();
    t1 = time.perf_counter()
    t2 = time.perf_counter();
    frame, _ = cam.snap();
    t3 = time.perf_counter()
    print("snap1:", t1 - t0, "snap2:", t3 - t2)
    print("read_mode:", cam.cam.get_read_mode())
    print("shape:", frame.shape)
    print("roi", cam.get_roi())


    cam.set_roi(2, 73, 5, 30,)
    cam.set_binning(32,21)
    # 3) snap one frame
    cam.snap();
    t0 = time.perf_counter();
    cam.snap();
    t1 = time.perf_counter()
    t2 = time.perf_counter();
    frame, _ = cam.snap();
    t3 = time.perf_counter()
    print("snap1:", t1 - t0, "snap2:", t3 - t2)
    print("read_mode:", cam.cam.get_read_mode())
    print("shape:", frame.shape)
    print("roi", cam.get_roi())




if __name__ == "__main__":
    test_ixon_camera()
