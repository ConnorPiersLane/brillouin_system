# vimba_startup_check.py
#
# Minimal Vimba health check — no GUI, no multiprocessing, no camera config.
# Run it directly on the acquisition machine:
#
#   python -m brillouin_system.devices.cameras.allied.vimba_startup_check
#
# It times each SDK step and prints where it blocks. Run it twice:
#   1) with the main GUI (and everything else) closed
#   2) with the main GUI running
# If (1) passes and (2) hangs at "Starting Vimba", the SDK startup is
# colliding with another running process. If both hang, the Vimba install
# or a registered transport layer is broken machine-wide (check whether
# Vimba Viewer starts, and reboot / repair the Vimba install).

import os
import time


def main():
    t0 = time.perf_counter()

    def stamp(msg):
        print(f"[{time.perf_counter() - t0:7.2f}s] {msg}", flush=True)

    stamp("Importing vimba package (loads VmbC.dll)...")
    from vimba import Vimba

    # The GenTL search paths tell us which vendors' transport layers
    # VmbStartup will try to load — a foreign/broken producer here is a
    # classic cause of an indefinite startup hang.
    for var in ("VIMBA_GENICAM_GENTL64_PATH", "GENICAM_GENTL64_PATH",
                "VIMBA_GENICAM_GENTL32_PATH", "GENICAM_GENTL32_PATH"):
        stamp(f"{var} = {os.environ.get(var, '<not set>')}")

    stamp("Entering Vimba.get_instance() (VmbStartup + transport layers)...")
    with Vimba.get_instance() as v:
        stamp("Vimba started.")

        stamp("Listing interfaces...")
        for inter in v.get_all_interfaces():
            stamp(f"  interface: {inter.get_id()}")

        stamp("Listing cameras...")
        cams = v.get_all_cameras()
        for cam in cams:
            stamp(f"  camera: {cam.get_id()}")
        if not cams:
            stamp("  NO CAMERAS FOUND")

        for cam in cams:
            stamp(f"Opening {cam.get_id()}...")
            try:
                with cam:
                    stamp(f"  opened OK, model: {cam.get_model()}")
            except Exception as e:
                stamp(f"  OPEN FAILED: {e!r}")

    stamp("Vimba shut down cleanly. All steps completed.")


if __name__ == "__main__":
    main()
