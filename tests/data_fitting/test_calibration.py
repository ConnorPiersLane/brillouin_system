import numpy as np
import matplotlib.pyplot as plt


from brillouinDAQ.utils.brillouin_spectrum_fitting import fit_calibration_curve, _linear_model

# === Test datasets ===
datasets = [
    {"px": [50.0, 60.0, 70.0],      "freq": [6.0, 5.0, 4.0],      "label": "Ideal Linear 1"},
    {"px": [58.0, 60.0, 62.0],      "freq": [4.9, 5.0, 5.1],      "label": "Centered 60 px"},
    {"px": [55.2, 60.1, 65.0],      "freq": [5.3, 5.0, 4.7],      "label": "Noisy Line"},
    {"px": [45.0, 60.0, 75.0],      "freq": [6.5, 5.0, 3.5],      "label": "Wider Spread"},
]

# === Plotting ===
plt.figure(figsize=(10, 6))
x_plot = np.linspace(40, 80, 200)

for data in datasets:
    px = np.array(data["px"])
    freq = np.array(data["freq"])
    label = data["label"]

    sd, fsr = fit_calibration_curve(px, freq)
    y_fit = _linear_model(x_plot, sd, fsr)

    # Plot original data points
    plt.plot(px, freq, 'o', label=f"{label} (SD={sd:.3f}, FSR={fsr:.2f})")

    # Plot fitted line
    plt.plot(x_plot, y_fit, '--', alpha=0.6)

plt.xlabel("Pixel Distance")
plt.ylabel("Frequency (GHz)")
plt.title("Brillouin Calibration Curve Fits")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
