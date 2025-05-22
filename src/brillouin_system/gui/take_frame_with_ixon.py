import matplotlib.pyplot as plt
import numpy as np
from brillouinDAQ.devices.cameras.for_brillouin_signal.ixonUltra import IxonUltra  # Adjust path if needed

def main():
    # --- Initialize Camera ---
    cam = IxonUltra(
        index=0,
        temperature="off",       # No cooling
        fan_mode="full",          # Full fan (fast operation)
        x_start=0, x_end=512,
    y_start = 0, y_end = 512,
    hbin=1,
        vbin=1,
        exposure_time=1,        # Exposure in seconds
        gain=1,
        advanced_gain_option=False,
        verbose=True
    )

    # --- Take a Frame ---
    frame = cam.snap()

    # --- Display Frame ---
    plt.figure(figsize=(8, 6))
    plt.imshow(frame, cmap="gray", aspect="equal", interpolation="none", origin="lower")
    plt.colorbar(label="Counts")
    plt.title("Ixon Ultra Captured Frame")
    plt.xlabel("Pixel (X)")
    plt.ylabel("Pixel (Y)")
    plt.show()

    # --- Close Camera ---
    cam.close()

if __name__ == "__main__":
    main()
