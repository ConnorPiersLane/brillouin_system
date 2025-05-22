import serial

class Microwave:
    def __init__(self, port="COM3", baudrate=115200, timeout=2.0):
        self.deviceName = "Synth"
        self.badCommand = b'[BADCOMMAND]\r\n'    # response if a command failed (b makes it into bytes)
        self.port = serial.Serial("COM3", 115200, timeout=10) #Change the COM PORT NUMBER to match your device
        if self.port.isOpen():    # make sure port is open
            self.port.write(b'*IDN?\n')   # send the standard SCPI identify command
            result = self.port.readline()
            print("[Microwave] Microwave source found: " + (result.strip()).decode('utf-8'))
        else:
            print('[Microwave] Could not open port')
        # Set initial RF power in dBm
        self.port.write(b'POWER +1.0dBm\n')
        self.port.write(b'POWER?\n')
        result = self.port.readline()
        print('[Microwave] RF power set to ' + (result.strip()).decode('utf-8'))
        # Set initial RF frequency in GHz
        self.port.write(b'FREQ:CW 5750MHz\n')
        self.port.write(b'FREQ:CW?\n')
        result = self.port.readline()
        print('[Microwave] RF frequency set to ' + (result.strip()).decode('utf-8'))
        # Enable RF output
        self.port.write(b'OUTP:STAT ON\n')
        self.port.write(b'OUTP:STAT?\n')
        result = self.port.readline()
        print('[Microwave] RF output is ' + (result.strip()).decode('utf-8'))
        self.runMode = 0    #0 is free running, 1 is scan

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        try:
            self.enable_output(False)
            self.port.close()
            print("[Microwave] Port closed.")
        except Exception:
            pass

    def _send(self, command: str):
        self.port.write((command + "\n").encode())

    def _query(self, command: str) -> str:
        self._send(command)
        return self.port.readline().decode().strip()

    def get_frequency(self) -> float:
        self.port.write(b'FREQ:CW?\n')  # try asking for signal generator setting
        result = self.port.readline()
        freq = float(result[:-4])*1e-9
        return freq

    def set_frequency(self, freq_ghz: float):
        #print('[Microwave] setFreq got called with f =', freq)
        freq_MHz = freq_ghz*1e3
        command = b'FREQ:CW %.1f' % freq_MHz + b'MHz\n'
        self.port.write(command)
        #print("[Microwave] RF frequency set to %.3f GHz" % freq)

    def get_power(self) -> float:
        self.port.write(b'POWER?\n')
        result = self.port.readline()
        power = float(result[:-5])
        return power

    def set_power(self, power_dbm: float):
        if power_dbm < 0:
            command = b'POWER -%.1f' % abs(power_dbm) + b'dBm\n'
        else:
            command = b'POWER +%.1f' % power_dbm + b'dBm\n'
        self.port.write(command)
        #print("[Microwave] Power set to %.1f dBm" % power_dbm)

    def enable_output(self, state: bool):
        self._send(f"OUTP:STAT {'ON' if state else 'OFF'}")
        print(f"[Microwave] RF output {'enabled' if state else 'disabled'}.")

    def is_output_enabled(self) -> bool:
        return self._query("OUTP:STAT?").strip() == "ON"


class MicrowaveDummy:
    def __init__(self):
        self.device_name = "MicrowaveDummy"
        self._frequency_ghz = 5.75
        self._power_dbm = 1.0
        self._output_enabled = True
        print(f"[{self.device_name}] Initialized at {self._frequency_ghz} GHz, {self._power_dbm} dBm")

    def shutdown(self):
        self._output_enabled = False
        print(f"[{self.device_name}] Shutdown (output off)")

    def get_frequency(self) -> float:
        print(f"[{self.device_name}] Current frequency: {self._frequency_ghz} GHz")
        return self._frequency_ghz

    def set_frequency(self, freq_ghz: float):
        self._frequency_ghz = freq_ghz
        print(f"[{self.device_name}] Frequency set to: {self._frequency_ghz} GHz")

    def get_power(self) -> float:
        print(f"[{self.device_name}] Current power: {self._power_dbm} dBm")
        return self._power_dbm

    def set_power(self, power_dbm: float):
        self._power_dbm = power_dbm
        print(f"[{self.device_name}] Power set to: {self._power_dbm} dBm")

    def enable_output(self, state: bool):
        self._output_enabled = state
        print(f"[{self.device_name}] Output {'enabled' if state else 'disabled'}")

    def is_output_enabled(self) -> bool:
        print(f"[{self.device_name}] Output is {'ON' if self._output_enabled else 'OFF'}")
        return self._output_enabled
