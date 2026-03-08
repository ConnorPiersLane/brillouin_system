import time
import numpy as np

from zaber_eye_lens_impl import ZaberEyeLens


def test_basic_motion(z):
    print("\n--- Basic Motion Test ---")

    print("Current position:", z.get_position())

    print("Moving +100 um")
    z.move_rel(100)
    print("Position:", z.get_position())

    print("Moving back to 0")
    z.move_abs(0)
    print("Position:", z.get_position())


def test_slew_guard(z):
    print("\n--- Slew Guard Test ---")

    z.start_position_log()

    print("Starting guarded slew (500 um/s for max 200 um)")
    z.start_slewing_guarded(500, 200)

    time.sleep(1)

    log = z.stop_position_log()

    print("Logged samples:", len(log.t_perf))
    print("Final position:", z.get_position())


def test_pvt_scan(z):
    print("\n--- PVT Scan Test ---")

    execution = z.start_linear_scan_pvt(
        speed_um_s=1000,
        distance_um=1000,
        dwell_s=0.05,
        accel_s=0.05,
        settle_s=0.01,
    )

    print("Launch window:", execution.launch_uncertainty_s)

    z.start_position_log()

    z.wait_for_pvt()

    log = z.stop_position_log()

    print("Log samples:", len(log.t_perf))

    # Compare measured vs model
    model_positions = [
        execution.model.position_at_perf(t)
        for t in log.t_perf
    ]

    error = np.array(log.z_um) - np.array(model_positions)

    print("Mean error (um):", np.mean(error))
    print("Max error (um):", np.max(np.abs(error)))


def main():
    z = ZaberEyeLens(port="COM5", axis_index=1)

    try:
        test_basic_motion(z)
        test_slew_guard(z)
        test_pvt_scan(z)

    finally:
        z.close()


if __name__ == "__main__":
    main()