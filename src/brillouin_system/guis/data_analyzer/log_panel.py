"""Shared log display for the data-analyzer windows.

One process-wide QtLogBridge fans every log line out to any number of
LogPanel widgets, so the manager and each open viewer can all carry their
own right-hand log column showing the same stream.

Everything funnels through ONE path to avoid duplicate lines: stdout and
stderr are teed into the bridge FIRST, and only then is the 'brillouin'
logger given its console fallback handler — that handler binds the teed
stderr, so logger output reaches the panel through the same tee that
catches plain print() calls (the analysis pipeline reports through
stdout). The original streams keep receiving everything; the panel is a
mirror, not a replacement.
"""
from __future__ import annotations

import sys

from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import QLabel, QPlainTextEdit, QPushButton, QHBoxLayout, QVBoxLayout, QWidget

from brillouin_system.logging_utils.logging_setup import enable_console_fallback
from brillouin_system.logging_utils.qt_log_bridge import QtLogBridge

_bridge: QtLogBridge | None = None
_streams_teed = False


def get_log_bridge() -> QtLogBridge:
    global _bridge
    if _bridge is None:
        _bridge = QtLogBridge()
    return _bridge


class _StreamTee:
    """File-like wrapper that mirrors complete lines into the bridge."""

    def __init__(self, original, bridge: QtLogBridge):
        self._orig = original
        self._bridge = bridge
        self._buf = ""

    def write(self, s: str) -> int:
        if self._orig is not None:
            try:
                self._orig.write(s)
            except Exception:
                pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                try:
                    self._bridge.message.emit(line)
                except RuntimeError:
                    pass
        return len(s)

    def flush(self):
        if self._orig is not None:
            try:
                self._orig.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._orig, name)


def install_analyzer_logging() -> QtLogBridge:
    """Route print() output and logger output into the shared bridge.

    Order matters: the streams are teed BEFORE the logger's console
    fallback is installed, so the fallback's StreamHandler binds the teed
    stderr and logger lines flow through the same single path as prints.
    Idempotent — safe to call from every window's constructor.
    """
    global _streams_teed
    bridge = get_log_bridge()
    if not _streams_teed:
        sys.stdout = _StreamTee(sys.stdout, bridge)
        sys.stderr = _StreamTee(sys.stderr, bridge)
        _streams_teed = True
    enable_console_fallback()
    return bridge


class LogPanel(QWidget):
    """Read-only, auto-scrolling log column fed by the shared bridge."""

    MAX_LINES = 5000

    def __init__(self, parent=None):
        super().__init__(parent)
        bridge = install_analyzer_logging()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.addWidget(QLabel("Log"))
        header.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(lambda: self.text.clear())
        header.addWidget(clear_btn)
        layout.addLayout(header)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(self.MAX_LINES)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(8)
        self.text.setFont(font)
        self.text.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; }"
        )
        layout.addWidget(self.text)

        bridge.message.connect(self.append_line)

    def append_line(self, line: str):
        self.text.appendPlainText(line)
        self.text.moveCursor(QTextCursor.End)
