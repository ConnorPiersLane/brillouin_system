import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Gaussian + pixel integration + sampling + reconstruction
#
# What is reconstructed?
#   h_rec(x) = F^{-1}{ S(f) on [-0.5, 0.5] }
#
# where h(x) = g * rect is the continuous pre-sampled,
# pixel-integrated signal.
#
# Then we can optionally estimate the original Gaussian by
# deconvolving the pixel aperture in the Nyquist band:
#
#   G_est(f) = H_rec(f) / sinc(f)
#
# with regularization.
# ------------------------------------------------------------

# -----------------------
# Parameters
# -----------------------
sigma = 0.5          # Gaussian std in pixels
dpx = 0.60           # subpixel shift of Gaussian center, in pixels
pixel_width = 1.0    # pixel width in pixel units
dx = 0.002           # fine continuous grid spacing
xmax = 6.0           # spatial extent
lam = 1e-3           # regularization for deconvolution
fplot = 2.0          # frequency plot extent

# -----------------------
# Grids
# -----------------------
x = np.arange(-xmax, xmax, dx)
N = len(x)

# Frequency grid in cycles/pixel
f = np.fft.fftfreq(N, d=dx)
f_shift = np.fft.fftshift(f)

# -----------------------
# Helper functions
# -----------------------
def sinc_cyc(ff):
    """sinc in cycles/pixel convention: sin(pi f)/(pi f)."""
    out = np.ones_like(ff, dtype=float)
    nz = np.abs(ff) > 1e-14
    out[nz] = np.sin(np.pi * ff[nz]) / (np.pi * ff[nz])
    return out

def gaussian(x, sigma, center=0.0):
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x - center)**2 / (2*sigma**2))

def rect_func(x, width=1.0):
    return np.where(np.abs(x) <= width/2, 1.0, 0.0)

# -----------------------
# Step 1: original Gaussian, shifted by dpx
# -----------------------
g = gaussian(x, sigma, center=dpx)

# -----------------------
# Step 2: pixel aperture (rect), centered at 0
# -----------------------
rect = rect_func(x, width=pixel_width)

# -----------------------
# Step 3: pixel-integrated continuous signal
# h = g * rect
# -----------------------
h = np.convolve(g, rect, mode='same') * dx

# -----------------------
# Step 4: sample at integer pixel centers
# -----------------------
n = np.arange(-5, 6, 1)
sample_idx = np.round((n - x[0]) / dx).astype(int)
s = h[sample_idx]

# -----------------------
# Reconstruct h(x) from the samples using sinc interpolation
# This is equivalent to inverse FT of S(f) on [-0.5, 0.5]
# -----------------------
h_rec_sinc = np.zeros_like(x)
for ni, si in zip(n, s):
    h_rec_sinc += si * sinc_cyc(x - ni)

# -----------------------
# Build sampled impulse train on the fine grid
# for explicit frequency-domain reconstruction
# -----------------------
s_train = np.zeros_like(x)
valid = (sample_idx >= 0) & (sample_idx < len(x))
s_train[sample_idx[valid]] = s[valid] / dx
# divide by dx so area of each discrete spike approximates sample weight

# FFT of sampled train
S_train = np.fft.fft(s_train)
S_train_shift = np.fft.fftshift(S_train)

# -----------------------
# Keep only the Nyquist band [-0.5, 0.5]
# This is the "remaining spectrum"
# -----------------------
nyq_mask = np.abs(f) <= 0.5
H_rec_band = np.zeros_like(S_train, dtype=complex)
H_rec_band[nyq_mask] = S_train[nyq_mask]

# Inverse FT -> reconstructed continuous h(x)
h_rec_fft = np.real(np.fft.ifft(H_rec_band))
# This h_rec_fft and h_rec_sinc should closely match.

# -----------------------
# Recover Gaussian estimate by deconvolving rect only:
# H(f) = G(f) * sinc(f)
# so G_est = H_rec / sinc(f), only in Nyquist band
# Use Tikhonov regularization.
# -----------------------
pix_mtf = sinc_cyc(f)
G_est = np.zeros_like(H_rec_band, dtype=complex)
G_est[nyq_mask] = H_rec_band[nyq_mask] * np.conj(pix_mtf[nyq_mask]) / (np.abs(pix_mtf[nyq_mask])**2 + lam)

g_rec = np.real(np.fft.ifft(G_est))

# -----------------------
# Analytic frequency-domain pieces for display
# -----------------------
G_analytic = np.exp(-2 * np.pi**2 * sigma**2 * f_shift**2) * np.exp(-1j * 2*np.pi * f_shift * dpx)
R_analytic = sinc_cyc(f_shift)
H_analytic = G_analytic * R_analytic

# Replicated spectrum after sampling
S_rep = np.zeros_like(f_shift, dtype=complex)
for k in range(-4, 5):
    fk = f_shift - k
    Gk = np.exp(-2 * np.pi**2 * sigma**2 * fk**2) * np.exp(-1j * 2*np.pi * fk * dpx)
    Rk = sinc_cyc(fk)
    S_rep += Gk * Rk

# -----------------------
# Diagnostics
# -----------------------
print("Interpretation:")
print("  h_rec(x) = inverse FT of sampled spectrum restricted to [-0.5, 0.5] cycles/pixel")
print("  g_rec(x) = deconvolution of h_rec by pixel sinc within [-0.5, 0.5]")
print()
print(f"sigma = {sigma:.3f} px")
print(f"dpx   = {dpx:.3f} px")
print(f"regularization lambda = {lam:.1e}")

# -----------------------
# Plotting
# -----------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Gaussian + Pixel Integration + Sampling + Reconstruction", fontsize=16)

# Step 1: original Gaussian + recovered Gaussian
ax = axes[0, 0]
ax.plot(x, g, label="Original Gaussian g(x)")
ax.plot(x, g_rec, "--", label="Recovered Gaussian estimate")
ax.axvline(dpx, color="k", linestyle=":", alpha=0.6, label=f"center = {dpx:.2f} px")
ax.set_title("Step 1: Original Gaussian and recovered Gaussian")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()

# Step 2: rect
ax = axes[0, 1]
ax.plot(x, rect, label="rect(x)")
ax.set_title("Step 2: Pixel aperture")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("Amplitude")
ax.set_xlim(-2, 2)
ax.set_ylim(-0.1, 1.2)
ax.grid(True)
ax.legend()

# Step 3: h = g * rect, plus reconstructed h
ax = axes[1, 0]
ax.plot(x, h, label="True h(x) = g * rect")
ax.plot(x, h_rec_fft, "--", label="Reconstructed h_rec(x)")
ax.set_title("Step 3: Pixel-integrated signal and reconstruction")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()

# Step 4: sampling
ax = axes[1, 1]
ax.plot(x, h, alpha=0.7, label="h(x)")
markerline, stemlines, baseline = ax.stem(n, s, linefmt='C1-', markerfmt='C1o', basefmt='k-')
plt.setp(stemlines, linewidth=1.5)
plt.setp(markerline, markersize=6)
ax.set_title("Step 4: Sampled signal at pixel centers")
ax.set_xlabel("x / pixel index")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()

# Step 5: analytic pre-sampled spectrum magnitude
ax = axes[2, 0]
mask_plot = np.abs(f_shift) <= fplot
ax.plot(f_shift[mask_plot], np.abs(G_analytic[mask_plot]), label="|G(f)| Gaussian")
ax.plot(f_shift[mask_plot], np.abs(R_analytic[mask_plot]), label="|sinc(f)| pixel")
ax.plot(f_shift[mask_plot], np.abs(H_analytic[mask_plot]), linewidth=2, label="|H(f)| = |G sinc|")
ax.axvline(-0.5, color='k', linestyle='--', alpha=0.6)
ax.axvline(0.5, color='k', linestyle='--', alpha=0.6, label="Nyquist band")
ax.set_title("Step 5: Pre-sampled spectrum magnitude")
ax.set_xlabel("Spatial frequency (cycles/pixel)")
ax.set_ylabel("Magnitude")
ax.grid(True)
ax.legend()

# Step 6: sampled spectrum replicas + Nyquist band kept
ax = axes[2, 1]
ax.plot(f_shift[mask_plot], np.abs(S_rep[mask_plot]), label="Replicated sampled spectrum |S(f)|")
band_show = np.zeros_like(f_shift, dtype=float)
band_show[np.abs(f_shift) <= 0.5] = np.abs(np.fft.fftshift(H_rec_band))[np.abs(f_shift) <= 0.5]
ax.plot(f_shift[mask_plot], band_show[mask_plot], "--", linewidth=2, label="Kept band for reconstruction")
ax.axvline(-0.5, color='k', linestyle='--', alpha=0.6)
ax.axvline(0.5, color='k', linestyle='--', alpha=0.6, label="Nyquist band")
ax.set_title("Step 6: Sampled spectrum and retained Nyquist interval")
ax.set_xlabel("Spatial frequency (cycles/pixel)")
ax.set_ylabel("Magnitude")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()