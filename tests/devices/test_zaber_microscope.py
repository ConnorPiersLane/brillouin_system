from time import sleep
from brillouin_system.devices.zaber_engines.zaber_microscope import ZaberMicroscope


def test_microscope():
    print("===== Zaber Microscope Integration Test =====")
    scope = None

    try:
        scope = ZaberMicroscope()

        print("\n🔹 Moving to absolute home position...")
        scope.move_axis_home()

        print("\n🔹 Moving X, Y, Z relatively by 100 µm...")
        scope.move_rel('x', 100)
        scope.move_rel('y', 100)
        scope.move_rel('z', 100)

        print("\n🔹 Reading current position:")
        pos = scope.get_zaber_position_class()
        print(f"  X = {pos.x:.1f} µm, Y = {pos.y:.1f} µm, Z = {pos.z:.1f} µm")

        print("\n🔹 Switching to Brightfield filter (index 1)...")
        scope.move_filter(1)
        sleep(1)

        print("🔹 Switching to Fluorescence filter (index 2)...")
        scope.move_filter(2)
        sleep(1)

        print("\n🔹 Testing LED channels...")
        print("  - White below ON")
        scope.led_white_below.on()
        sleep(1)
        scope.led_white_below.off()

        print("  - Blue 385 below ON")
        scope.led_blue_385_below.on()
        sleep(1)
        scope.led_blue_385_below.off()

        print("  - Red 625 below ON")
        scope.led_red_625_below.on()
        sleep(1)
        scope.led_red_625_below.off()

        print("  - White top ON")
        scope.led_white_top.on()
        sleep(1)
        scope.led_white_top.off()

    except Exception as e:
        print(f"\n❌ Test Error: {e}")
    finally:
        if scope:
            print("\n🔻 Shutting down microscope...")
            scope.shutdown()


if __name__ == "__main__":
    test_microscope()
