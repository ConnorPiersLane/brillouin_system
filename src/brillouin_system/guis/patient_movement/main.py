"""
Entry point for the Patient Movement Tracker GUI.

Desk mode (default): simulated NI + eye lens with a moving simulated cornea,
dummy rig stage, dummy eye-tracker cameras — the whole workflow (find plane,
track, laser calib buttons) is exercisable without hardware.

    python -m brillouin_system.guis.patient_movement.main

On the instrument:

    python -m brillouin_system.guis.patient_movement.main --real

Use --real-cameras to run real Allied Vision cameras with simulated NI/Zaber
(or --real, which implies real cameras too).
"""

from __future__ import annotations

import argparse
import sys

from PyQt5.QtWidgets import QApplication

from brillouin_system.guis.patient_movement.pm_frontend import PmFrontend


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient Movement Tracker")
    parser.add_argument("--real", action="store_true",
                        help="use real NI/Zaber hardware AND real cameras")
    parser.add_argument("--real-cameras", action="store_true",
                        help="use real Allied Vision cameras (NI/Zaber stay simulated)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    w = PmFrontend(
        use_backend_dummy=not args.real,
        use_eye_tracker_dummy=not (args.real or args.real_cameras),
    )
    w.resize(1500, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
