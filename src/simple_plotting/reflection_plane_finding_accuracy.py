import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Replace these with your data
# -----------------------------
slewing_estimates = [
    16745.949500000002,
17442.893750000003,
16730.80475,
17444.22725,
16730.61425,
17443.036625,
16729.9475,
17452.371124999998,
16730.280875,
]

max_refine_estimates = [
    16735.949500000002,
17452.893750000003,
16740.80475,
17449.22725,
16745.61425,
17448.036625,
16744.9475,
17442.371124999998,
16740.280875,
]

parable_fit_estimates = [
    16736.962018409427,
17450.521955128206,
16742.599904185023,
17448.95327739726,
16744.65757129964,
17447.192030405407,
16743.749275147926,
17444.1411979927,
16739.218783496734,
]

fine_peak_estimates = [
    16741.95,
17444.89,
16741.80,
17443.23,
16740.61,
17443.04,
16738.95,
17444.37,
16741.28,
]

# -----------------------------
# Pack data
# -----------------------------
labels = [
    "Slewing estimate",
    "Max refine",
    "Parabolic fit",
    "Fine scan peak"
]

data = [
    slewing_estimates,
    max_refine_estimates,
    parable_fit_estimates,
    fine_peak_estimates
]

means = [np.mean(d) for d in data]
stds  = [np.std(d, ddof=1) for d in data]  # sample std

# -----------------------------
# Plot
# -----------------------------
x = np.arange(len(labels))

plt.figure(figsize=(7, 5))
plt.errorbar(
    x,
    means,
    yerr=stds,
    fmt='o',
    capsize=6,
    elinewidth=2,
    markersize=8
)

plt.xticks(x, labels, rotation=20)
plt.ylabel("Z position (µm)")
plt.title("Reflection plane estimates\nMean ± 1σ over runs")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()