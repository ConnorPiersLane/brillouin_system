"""
Entry point for the Patient Movement Tracker GUI.

Set the flags below (same style as hi_frontend), then run this file
(e.g. from PyCharm).

Dummy mode: simulated NI + eye lens with a moving simulated cornea, dummy
rig stage, dummy eye-tracker cameras — the whole workflow (find plane,
track, laser positioning) is exercisable without hardware.
"""

from __future__ import annotations

import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from brillouin_system.guis.patient_movement.pm_frontend import PmFrontend
from brillouin_system.logging_utils.logging_setup import install_crash_hooks, start_logging

use_backend_dummy = False       # NI + Zaber eye lens + rig stage
use_eye_tracker_dummy = False   # Allied Vision cameras / eye-tracker worker


def main():
    # Set rounding policy before constructing QApplication (Qt >= 5.14)
    try:
        if hasattr(QtWidgets.QApplication, "setHighDpiScaleFactorRoundingPolicy"):
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor
            )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        * { font-size: 8pt; }
    """)
    viewer = PmFrontend(
        use_backend_dummy=use_backend_dummy,
        use_eye_tracker_dummy=use_eye_tracker_dummy,
    )
    viewer.resize(1500, 900)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # IMPORTANT: On Windows, spawn the logging writer process ONLY here.
    start_logging()
    install_crash_hooks()

    main()
