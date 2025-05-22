
from ctypes import *


ERROR_CODE = {
    0: 'PI_NO_ERROR',
    1: 'PI_DEVICE_NOT_FOUND',
    2: 'PI_OBJECT_NOT_FOUND',
    3: 'PI_CANNOT_CREATE_OBJECT',
    4: 'PI_INVALID_DEVICE_HANDLE',
    5: 'PI_READ_TIMEOUT',
    6: 'PI_READ_THREAD_ABANDONED',
    7: 'PI_READ_FAILED',
    8: 'PI_INVALID_PARAMETER',
    9: 'PI_WRITE_FAILED'
}

class ShutterDummy:
    def __init__(self, usb_code: int):
        self.usb_code = usb_code
        self._state = False  # False = closed, True = open
        print(f"[ShutterDummy] Initialized dummy shutter (USB code: {usb_code})")

    def open(self):
        self._state = True
        print(f"[ShutterDummy] Shutter {self.usb_code} opened.")

    def close(self):
        self._state = False
        print(f"[ShutterDummy] Shutter {self.usb_code} closed.")

    def get_state(self):
        print(f"[ShutterDummy] Shutter {self.usb_code} state: {'open' if self._state else 'closed'}")
        return self._state

    def shutdown(self):
        print(f"[ShutterDummy] Shutter {self.usb_code} shut down.")

class Shutter:
    def __init__(self, usb_code):
        self.usb_code = usb_code
        self.dll = WinDLL(r'C:\Program Files\PiUsbSDK\bin\x64\PiUsb')
        self.dll.piConnectShutter.restype = c_longlong
        self._connect()

    def _connect(self):
        error = c_int()
        self.handle = c_longlong(self.dll.piConnectShutter(byref(error), self.usb_code))
        if error.value != 0:
            print(f"[Shutter] Failed to connect to shutter {self.usb_code}: {ERROR_CODE.get(error.value, 'Unknown error')}")
        else:
            print(f"[Shutter] Connected to shutter {self.usb_code}")

    def set_state(self, state: bool):
        """Open (True) or Close (False) the shutter."""
        val = 1 if state else 0
        error = self.dll.piSetShutterState(val, self.handle)
        if error != 0:
            print(f"[Shutter] Failed to set state {val}: {ERROR_CODE.get(error, 'Unknown error')}")

    def get_state(self):
        state = c_int()
        error = self.dll.piGetShutterState(byref(state), self.handle)
        if error != 0:
            print(f"[Shutter] Failed to get state: {ERROR_CODE.get(error, 'Unknown error')}")
            return None
        return bool(state.value)

    def open(self):
        self.set_state(True)

    def close(self):
        self.set_state(False)

    def shutdown(self):
        if self.handle:
            self.dll.piDisconnectShutter(self.handle)
            print(f"[Shutter] Disconnected shutter {self.usb_code}")



class ShutterManager:
    """Convenience manager to group multiple shutters logically."""
    def __init__(self, sample: str):
        """

        Args:
            sample: "microscope" or "human_interface"
        """
        # Consistently defined shutter codes
        # self.codes = {
        #     "objective": 384,
        #     "reference": 338,
        #     "microscope": 339,
        #     "human_interface": 314
        # }

        self.sample_type = sample.lower()

        if self.sample_type == 'microscope':
            self.sample = Shutter(339)
        elif self.sample_type == 'human_interface':
            self.sample = Shutter(314)
        else:
            raise ValueError(f"Incorrect parameter input sample={sample}")

        self.objective = Shutter(384)
        self.reference = Shutter(338)

    def close_all(self):
        self.objective.close()
        self.reference.close()
        self.sample.close()

    def change_to_reference(self):
        self.objective.close()
        self.reference.open()

    def change_to_objective(self):
        self.reference.close()
        self.objective.open()

    def shutdown_all(self):
        self.objective.shutdown()
        self.reference.shutdown()
        self.sample.shutdown()
        print("[ShutterController] All shutters shut down.")


class ShutterManagerDummy:
    def __init__(self, sample: str):
        if sample.lower() not in ['microscope', 'human_interface']:
            raise ValueError(f"[DummyShutterController] Invalid sample type: {sample}")

        self.sample_name = sample.lower()

        # Attach dummy shutters
        self.sample = ShutterDummy(314 if self.sample_name == 'human_interface' else 339)
        self.objective = ShutterDummy(384)
        self.reference = ShutterDummy(338)

        print(f"[DummyShutterController] Initialized with sample='{self.sample_name}'")

    def close_all(self):
        self.objective.close()
        self.reference.close()
        self.sample.close()

    def change_to_reference(self):
        self.objective.close()
        self.reference.open()

    def change_to_objective(self):
        self.reference.close()
        self.objective.open()

    def shutdown_all(self):
        self.objective.shutdown()
        self.reference.shutdown()
        self.sample.shutdown()
        print("[ShutterController] All shutters shut down.")
