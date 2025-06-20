import threading
import queue
from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion.ascii import Axis
from zaber_motion.ascii import Device

Library.enable_device_db_store()

class ZaberMicroscope:
    def __init__(self, port="COM4"):
        self.command_queue = queue.Queue()
        self._last_position = [0.0, 0.0, 0.0]  # x, y, z in um

        # Microstep size in um (from original code)
        self.xy_microstep_size = 0.15625
        self.z_microstep_size = 0.001

        self.home_pos = {
            'x': 60000.0,
            'y': 50000.0,
            'z': 6990.0
        }

        self._connect(port)
        self._initialize_devices()
        self._home_all()
        self._read_max_currents()
        self._set_initial_lighting()

    def _connect(self, port):
        self.connection = Connection.open_serial_port(port)
        print("[ZaberDevice] Connected to port:", port)

    def _initialize_devices(self):
        self.devices = {
            'xy': self.connection.detect_devices()[2],  # Device 3
            'z': self.connection.detect_devices()[3],   # Device 4
            'light': self.connection.detect_devices()[1],  # Device 2
            'filter': self.connection.detect_devices()[4]  # Device 5
        }

        self.axes = {
            'x': self.devices['xy'].get_axis(2),
            'y': self.devices['xy'].get_axis(1),
            'z': self.devices['z'].get_axis(1)
        }

        self.lights = {
            'white': self.devices['light'].get_axis(1),
            'blue': self.devices['light'].get_axis(2),
            'red': self.devices['light'].get_axis(3),
            'trans': self.devices['light'].get_axis(4)
        }

    def _home_all(self):
        for axis in self.axes.values():
            axis.home().wait()
        self.devices['filter'].home().wait()
        print("[ZaberDevice] All axes and filter wheel homed")

    def _read_max_currents(self):
        self.max_currents = {}
        for color, lamp in self.lights.items():
            reply = lamp.send("get lamp.current.max").data
            self.max_currents[color] = float(reply)
            print(f"[ZaberDevice] {color} max current: {reply} A")

    def _set_initial_lighting(self):
        self.set_light('trans', True)
        for color in ['white', 'blue', 'red']:
            self.set_light(color, False)

    def shutdown(self):
        with self.lock:
            self.connection.close()
        print("[ZaberDevice] Connection closed")


    def move_abs(self, axis, position_um):
        step_size = self.xy_microstep_size if axis in ['x', 'y'] else self.z_microstep_size
        steps = int(position_um / step_size)
        with self.lock:
            self.axes[axis].move_absolute(steps).wait()
        self.update_position(axis)

    def move_rel(self, axis, distance_um):
        step_size = self.xy_microstep_size if axis in ['x', 'y'] else self.z_microstep_size
        steps = int(distance_um / step_size)
        with self.lock:
            self.axes[axis].move_relative(steps).wait()
        self.update_position(axis)

    def move_home(self, axis=None):
        if axis:
            self.move_abs(axis, self.home_pos[axis])
        else:
            for ax in ['x', 'y', 'z']:
                self.move_abs(ax, self.home_pos[ax])

    def update_position(self, axis=None):
        with self.lock:
            if axis:
                pos = self.axes[axis].get_position()
                idx = ['x', 'y', 'z'].index(axis)
                step_size = self.xy_microstep_size if axis in ['x', 'y'] else self.z_microstep_size
                self._last_position[idx] = pos * step_size
            else:
                for i, ax in enumerate(['x', 'y', 'z']):
                    pos = self.axes[ax].get_position()
                    step_size = self.xy_microstep_size if ax in ['x', 'y'] else self.z_microstep_size
                    self._last_position[i] = pos * step_size
        return self._last_position

    def set_light(self, color, on=True):
        with self.lock:
            command = "lamp on" if on else "lamp off"
            reply = self.lights[color].send(command)
            if reply.reply_flag != "OK":
                print(f"[ZaberDevice] Failed to set {color} light {'on' if on else 'off'}")

    def set_power(self, color, percent):
        current = percent * self.max_currents[color] / 100.0
        with self.lock:
            reply = self.lights[color].send(f"set lamp.current {current}")
            if reply.reply_flag != "OK":
                print(f"[ZaberDevice] Failed to set {color} power")

    def get_power(self, color):
        with self.lock:
            reply = self.lights[color].send("get lamp.current")
            current = float(reply.data)
        return 100 * current / self.max_currents[color]

    def move_filter(self, index=1):
        with self.lock:
            reply = self.devices['filter'].send(f"move index {index}")
            if reply.reply_flag != "OK":
                print(f"[ZaberDevice] Failed to move filter wheel to index {index}")

    def set_async(self, method, *args):
        self.command_queue.put((method, *args))
