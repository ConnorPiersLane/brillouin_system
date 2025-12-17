from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra


def test_ixon_camera():
    print("=== Starting IxonUltra camera test (no cooling) ===")

    cam = IxonUltra(index=0, temperature="off", fan_mode="full")
    cam.set_roi(0, 128, 0, 512,)
    cam.set_binning(4,1)
    cam.cam.set_read_mode('fvb')
    # 3) snap one frame
    frame,_ = cam.snap()
    print("read_mode:", cam.cam.get_read_mode())
    print("shape:", frame.shape)
    print("roi", cam.get_roi())


    cam.set_roi(0, 71, 0, 20,)
    cam.set_binning(32,21)
    # 3) snap one frame
    frame,_ = cam.snap()
    print("read_mode:", cam.cam.get_read_mode())
    print("shape:", frame.shape)
    print("roi", cam.get_roi())



if __name__ == "__main__":
    test_ixon_camera()
