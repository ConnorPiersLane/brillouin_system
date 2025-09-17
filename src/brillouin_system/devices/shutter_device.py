
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

# TODO: Test this
if __name__ == "__main__":
    import time
    def measure(shutter, n=20):
        latencies = []
        for _ in range(n):
            t0 = time.perf_counter()
            shutter.open()
            while shutter.get_state() is not True:
                pass
            t1 = time.perf_counter()
            shutter.close()
            while shutter.get_state() is not False:
                pass
            t2 = time.perf_counter()
            latencies.append((t1 - t0, t2 - t1))  # (open_time, close_time)
        return latencies
    latencies = measure(Shutter(314))
    print(latencies)
# Result:
# [Shutter] Connected to shutter 314
# [(0.07668190007098019, 0.06410329998470843), (0.0640064999461174, 0.06399499997496605), (0.06393800000660121, 0.0639837000053376), (0.06397829996421933, 0.06402350007556379), (0.06404119986109436, 0.06401400011964142), (0.06400109990499914, 0.06387570011429489), (0.06403469992801547, 0.06403820007108152), (0.06393509986810386, 0.06407560012303293), (0.06399799999780953, 0.06400209991261363), (0.06390770012512803, 0.06408299994654953), (0.06390449986793101, 0.06408090004697442), (0.06391449994407594, 0.0640180001500994), (0.06406980007886887, 0.0640024000313133), (0.06398750003427267, 0.06393379997462034), (0.06406470015645027, 0.0639286998193711), (0.06406250013969839, 0.06402459996752441), (0.06396730011329055, 0.064007299952209), (0.06401459989137948, 0.06396650010719895), (0.06390960002318025, 0.06415320001542568), (0.06391440005972981, 0.06401659990660846)]
#
# --> 64ms
