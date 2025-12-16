from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QCoreApplication
from PyQt5 import QtCore

from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
from brillouin_system.eye_tracker.eye_tracker_proxy import EyeTrackerProxy


class EyeTrackerController(QObject):
    frames_ready = QtCore.pyqtSignal(object, object, dict)  # left, right, rendered, meta
    log_message = QtCore.pyqtSignal(str)

    def __init__(self, use_dummy: bool):
        super().__init__()
        self.proxy = EyeTrackerProxy(use_dummy=use_dummy)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._running = False

    @QtCore.pyqtSlot()
    def start(self):
        try:
            self.proxy.start()  # launches worker process and attaches rings
            self._running = True
            self._timer.start(15)  # ~60–70 Hz polling
            self.log_message.emit("EyeTracker started.")
        except Exception as e:
            self.log_message.emit(f"Failed to start EyeTracker: {e}")
            raise e

    @QtCore.pyqtSlot()
    def stop(self):
        self._timer.stop()
        self._running = False
        try:
            self.proxy.stop_streaming()
        except Exception:
            pass

    @QtCore.pyqtSlot()
    def shutdown(self):
        self.stop()
        try:
            self.proxy.shutdown()
        except Exception:
            pass
        self.log_message.emit("EyeTracker shutdown complete.")


    @QtCore.pyqtSlot(object)
    def send_config(self, config: EyeTrackerConfig):
        self.proxy.set_et_config(config)
        self.log_message.emit("Send new config to EyeTracker")

    def _poll(self):
        if not self._running:
            return
        try:
            result = self.proxy.get_latest(timeout=0)
            if result is None:
                return
            left, right, meta = result
            self.frames_ready.emit(left, right, meta)
        except Exception as e:
            self.log_message.emit(f"EyeTracker polling error: {e}")
            raise e
