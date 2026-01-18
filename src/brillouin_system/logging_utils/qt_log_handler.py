# qt_log_handler.py
import logging

from brillouin_system.logging_utils.qt_log_bridge import QtLogBridge


class QtTextEditHandler(logging.Handler):
    def __init__(self, bridge: QtLogBridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record):
        try:
            self.bridge.message.emit(self.format(record))
        except Exception:
            self.handleError(record)
