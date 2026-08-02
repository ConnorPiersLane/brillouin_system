import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1D Gaussian signal + pixel integration (rect) + sampling (comb)
# Plots each step from the derivation:
#   1) Gaussian PSF g(x)
#   2) Pixel aperture rect(x)
#   3) Integrated signal h(x) = g * rect
#   4) Sampled signal at pixel centers
#   5) Frequency responses: G(f), sinc(f), H(f)=G(f)*sinc(f)
#   6) Spectrum replicas after sampling
# ------------------------------------------------------------

# -----------------------
# Parameters
# -----------------------
sigma = 0.5          # Gaussian std in pixels
pixel_width = 1.0    # pixel pitch / aperture width, in pixel units
dx = 0.002           # fine continuous-space grid
xmax = 6.0           # spatial extent for plotting
fmax = 2.5           # frequency extent for plotting (cycles/pixel)

# Continuous spatial grid
x = np.arange(-xmax, xmax + dx, dx)

# -----------------------
# Step 1: Gaussian signal
# -----------------------
g = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))

# -----------------------
# Step 2: Pixel aperture = rect(x)
# width = 1 pixel, centered at 0
# -----------------------
rect = np.where(np.abs(x) <= pixel_width / 2, 1.0, 0.0)

# -----------------------
# Step 3: Pixel integration
# h(x) = g * rect
# Continuous convolution approximated numerically
# Multiply by dx because np.convolve approximates sum, not integral
# -----------------------
h = np.convolve(g, rect, mode='same') * dx

# Optional: equivalent Gaussian approximation after pixel integration
sigma_eff = np.sqrt(sigma**2 + pixel_width**2 / 12)

# -----------------------
# Step 4: Sample at pixel centers
# s[n] = h(n), n integer
# -----------------------
n = np.arange(-5, 6, 1)  # sample locations in pixels
sample_idx = np.round((n - x[0]) / dx).astype(int)
s = h[sample_idx]

# Exact pixel-integrated samples using Gaussian CDF / erf would match closely,
# but here we directly sample h(x), which is already the integrated signal.

# -----------------------
# Step 5: Frequency-domain expressions
# Frequency in cycles/pixel
# -----------------------
f = np.linspace(-fmax, fmax, 4000)

def sinc_cyc(ff):
    # sinc in cycles/pixel convention: sin(pi f)/(pi f)
    out = np.ones_like(ff)
    nz = ff != 0
    out[nz] = np.sin(np.pi * ff[nz]) / (np.pi * ff[nz])
    return out

G = np.exp(-2 * np.pi**2 * sigma**2 * f**2)
R = sinc_cyc(f)
H = G * R

# -----------------------
# Step 6: Spectrum after sampling
# S(f) = sum_k H(f-k)
# Show a few replicas
# -----------------------
S = np.zeros_like(f)
kmin, kmax = -4, 4
for k in range(kmin, kmax + 1):
    fk = f - k
    S += np.exp(-2 * np.pi**2 * sigma**2 * fk**2) * sinc_cyc(fk)

# -----------------------
# Some useful numbers
# -----------------------
fwhm_g = 2 * np.sqrt(2 * np.log(2)) * sigma
fwhm_eff = 2 * np.sqrt(2 * np.log(2)) * sigma_eff
nyquist = 0.5
H_nyq = np.exp(-2 * np.pi**2 * sigma**2 * nyquist**2) * sinc_cyc(np.array([nyquist]))[0]

print(f"Input Gaussian sigma       = {sigma:.4f} px")
print(f"Input Gaussian FWHM        = {fwhm_g:.4f} px")
print(f"Effective sigma after rect = {sigma_eff:.4f} px")
print(f"Effective FWHM after rect  = {fwhm_eff:.4f} px")
print(f"Presampled MTF at Nyquist  = {H_nyq:.4f}")

# -----------------------
# Plotting
# -----------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Gaussian + Pixel Integration + Sampling", fontsize=16)

# 1) Gaussian
ax = axes[0, 0]
ax.plot(x, g, label=r"$g(x)$ Gaussian")
ax.set_title("Step 1: Continuous Gaussian signal")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()

# 2) Rect aperture
ax = axes[0, 1]
ax.plot(x, rect, label=r"$\mathrm{rect}(x)$ pixel aperture")
ax.set_title("Step 2: Pixel aperture (rect)")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("Amplitude")
ax.set_xlim(-2, 2)
ax.set_ylim(-0.1, 1.2)
ax.grid(True)
ax.legend()

# 3) Convolution result
ax = axes[1, 0]
ax.plot(x, g, '--', alpha=0.7, label=r"$g(x)$")
ax.plot(x, h, label=r"$h(x)=g * \mathrm{rect}$")
# equivalent Gaussian approximation
g_eff = (1 / (np.sqrt(2 * np.pi) * sigma_eff)) * np.exp(-x**2 / (2 * sigma_eff**2))
ax.plot(x, g_eff, ':', label=rf"Gaussian approx, $\sigma_{{eff}}={sigma_eff:.3f}$")
ax.set_title("Step 3: Pixel-integrated continuous signal")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()

# 4) Sampled signal
ax = axes[1, 1]
ax.plot(x, h, alpha=0.7, label=r"$h(x)$ continuous integrated signal")
ax.stem(n, s, linefmt='C1-', markerfmt='C1o', basefmt='k-', label=r"$s[n]=h(n)$")
ax.set_title("Step 4: Sampled at pixel centers")
ax.set_xlabel("Pixel index / position")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()

# 5) Presampled frequency response
ax = axes[2, 0]
ax.plot(f, G, label=r"$G(f)=e^{-2\pi^2\sigma^2 f^2}$")
ax.plot(f, R, label=r"$\mathrm{sinc}(f)$")
ax.plot(f, H, linewidth=2, label=r"$H(f)=G(f)\,\mathrm{sinc}(f)$")
ax.axvline(0.5, color='k', linestyle='--', alpha=0.6, label="Nyquist")
ax.axvline(-0.5, color='k', linestyle='--', alpha=0.6)
ax.set_title("Step 5: Presampled frequency response")
ax.set_xlabel("Spatial frequency (cycles/pixel)")
ax.set_ylabel("Magnitude")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.25, 1.05)
ax.grid(True)
ax.legend()

# 6) Spectrum replicas after sampling
ax = axes[2, 1]
ax.plot(f, H, '--', alpha=0.7, label="Presampled H(f)")
for k in range(kmin, kmax + 1):
    fk = f - k
    Hk = np.exp(-2 * np.pi**2 * sigma**2 * fk**2) * sinc_cyc(fk)
    ax.plot(f, Hk, alpha=0.35)
ax.plot(f, S, linewidth=2, label=r"$S(f)=\sum_k H(f-k)$")
ax.axvline(0.5, color='k', linestyle='--', alpha=0.6, label="Nyquist")
ax.axvline(-0.5, color='k', linestyle='--', alpha=0.6)
ax.set_title("Step 6: Replicated spectrum after sampling")
ax.set_xlabel("Spatial frequency (cycles/pixel)")
ax.set_ylabel("Magnitude")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.1, 1.2)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()