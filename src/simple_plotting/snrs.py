import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FormatStrFormatter

# --- Store data in lists ---
noise_std = [9.3, 6.4, 5.3, 4.6, 4.0, 4.2, 2.9, 3.2, 2.9, 3.1]  # MHz
photons   = [2089, 4069, 5956, 7967, 10579, 12290, 14183, 16332, 18680, 21194]
energy    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # mJ

# --- Plot Noise vs. Photons ---
plt.figure(figsize=(6, 4))
plt.loglog(photons, noise_std, 'o-', linewidth=2)
plt.xlabel("Photons (N)")
plt.ylabel("Noise Std [MHz]")
plt.title("Noise vs. Photons")
plt.grid(True, which="both", ls="--", alpha=0.7)

ax = plt.gca()
# Show all ticks and format them
ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=range(1, 10), numticks=20))
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=range(1, 10), numticks=20))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

# Rotate x-axis tick labels for readability
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# --- Plot Noise vs. Energy ---
plt.figure(figsize=(6, 4))
plt.loglog(energy, noise_std, 'o-', linewidth=2)
plt.xlabel("Energy [mJ]")
plt.ylabel("Noise Std [MHz]")
plt.title("Noise vs. Energy")
plt.grid(True, which="both", ls="--", alpha=0.7)

ax = plt.gca()
ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=range(1, 10), numticks=20))
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=range(1, 10), numticks=20))
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

plt.tight_layout()
plt.show()
