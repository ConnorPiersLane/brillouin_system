import numpy as np
import matplotlib.pyplot as plt

# timestamps (or whatever values you gave)
values = np.array([
8415.95210532, 8415.96823827, 8415.98423975, 8416.00018917, 8416.01611715,
8416.03208597, 8416.06154252, 8416.0798685,  8416.09591215, 8416.11188985,
8416.12787032, 8416.1438224,  8416.15991417, 8416.17588267, 8416.19186672,
8416.20782472, 8416.22373202, 8416.23969988, 8416.25576092, 8416.27189408,
8416.28788077, 8416.30384293
])

# example Zaber positions (replace with your real logged positions)
zaber_lens_z_um = np.fromstring("""
12499.990875 12499.990875 12482.750625 12421.885875 12342.39975
 12262.38975  12118.895625 12023.88375  11943.3975   11863.911375
 11783.3775   11703.891375 11620.738125 11588.734125 11586.448125
 11586.448125 11586.448125 11586.448125 11586.448125 11586.448125
 11586.448125 11586.448125
""", sep=' ')

# differences
diffs = np.diff(values)

# midpoint positions (for plotting)
z_mid = 0.5 * (zaber_lens_z_um[:-1] + zaber_lens_z_um[1:])

print("Differences:", diffs)

plt.figure()

plt.plot(zaber_lens_z_um[:-1], diffs * 1000, marker="o")  # convert to ms if these are timestamps
plt.xlabel("Zaber position (µm)")
plt.ylabel("Δ time (ms)")
plt.title("Time difference vs Zaber position")
plt.grid(True)

plt.show()