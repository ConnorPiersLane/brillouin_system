# qt_log_bridge.py
from PyQt5.QtCore import QObject, pyqtSignal
class QtLogBridge(QObject):
    message = pyqtSignal(str)
