from brillouinDAQ.config.enums import SystemMode
from time import sleep
from brillouinDAQ.devices.ShutterDevice import ShutterDevice


def test_shutter_device():
    print("Initializing ShutterDevice...")

    try:
        device = ShutterDevice(SystemMode.MACRO)

        print("Default State:", device.getShutterState())

        # Test setting the shutters to SAMPLE_STATE
        print("\nSetting to SAMPLE_STATE...")
        device.setShutterState(ShutterDevice.SAMPLE_STATE)
        sleep(1)
        print("Current State:", device.getShutterState())

        # Test setting the shutters to REFERENCE_STATE
        print("\nSetting to REFERENCE_STATE...")
        device.setShutterState(ShutterDevice.REFERENCE_STATE)
        sleep(1)
        print("Current State:", device.getShutterState())

        # Test setting the shutters to CLOSED_STATE
        print("\nClosing shutters...")
        device.setShutterState(ShutterDevice.CLOSED_STATE)
        sleep(1)
        print("Current State:", device.getShutterState())

        # Test setting the shutters to OPEN_STATE
        print("\nOpening shutters...")
        device.setShutterState(ShutterDevice.OPEN_STATE)
        sleep(1)
        print("Current State:", device.getShutterState())

        # Test BG shutter operations
        print("\nOpening BG Shutter...")
        device.openBGshutter()
        sleep(1)

        print("\nClosing BG Shutter...")
        device.closeBGshutter()
        sleep(1)

        # Test setting back to SAMPLE_STATE
        print("\nResetting to SAMPLE_STATE...")
        device.setShutterState(ShutterDevice.SAMPLE_STATE)
        sleep(1)
        print("Final State:", device.getShutterState())

    except Exception as e:
        print(f"Error during test: {e}")

    finally:
        print("\nShutting down device...")
        device.shutdown()
        print("Test complete.")

if __name__ == "__main__":
    test_shutter_device()
