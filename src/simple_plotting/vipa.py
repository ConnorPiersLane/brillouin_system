import numpy as np
import matplotlib.pyplot as plt

# ==== Parameters ====
FSR_GHz = 20.0          # Free Spectral Range (GHz)
finesse = 30.0          # Cavity finesse
B_shift_GHz = 4      # Brillouin frequency shift (GHz)
brillouin_rel_amp = 0.18    # Sideband amplitude relative to Airy peak
brillouin_FWHM_GHz = 0.06   # Brillouin linewidth (GHz)

# ---- Derived cavity parameters ----
pi = np.pi
F = finesse
# Solve for sqrt(R) from finesse = pi*sqrt(R)/(1-R)
x = (-pi + np.sqrt(pi**2 + 4*F**2)) / (2*F)
R = x**2
Fcoeff = 4*R/(1 - R)**2  # coefficient of finesse

def airy_T(f_GHz):
    """Airy transmission (normalized) at frequency f_GHz."""
    delta = 2*np.pi * (f_GHz / FSR_GHz)
    return 1.0 / (1.0 + Fcoeff * np.sin(0.5 * delta)**2)

def lorentz(f, f0, fwhm):
    """Normalized Lorentzian line shape."""
    return 1.0 / (1.0 + ((f - f0) / (0.5 * fwhm))**2)

# Frequency axis covering two resonances
f = np.linspace(-5, FSR_GHz + 5, 4000)

# Airy transmission baseline
T_airy = airy_T(f)

# Add Brillouin sidebands around each resonance (Stokes & anti-Stokes)
centers = [0.0, FSR_GHz]
brillouin = np.zeros_like(f)
for c in centers:
    brillouin += brillouin_rel_amp * lorentz(f, c - B_shift_GHz, brillouin_FWHM_GHz)
    brillouin += brillouin_rel_amp * lorentz(f, c + B_shift_GHz, brillouin_FWHM_GHz)

total_spec = T_airy + brillouin

# ---- Plot ----
plt.figure(figsize=(8, 4.5))
plt.plot(f, total_spec, )#label="Total (Airy + Brillouin)")
plt.plot(f, T_airy, "--", )#label="Airy only")
plt.plot(f, brillouin, ":", )#label="Brillouin sidebands")

# Annotate resonances
for c in centers:
    plt.axvline(c, linestyle="--", alpha=0.3)
    # plt.text(c, 0.92, "Airy peak", ha="center", va="top")

# plt.text(-B_shift_GHz, 0.6, "Stokes", ha="center")
# plt.text(+B_shift_GHz, 0.6, "anti-Stokes", ha="center")

plt.xlabel("Frequency (GHz)")
plt.ylabel("Normalized intensity (a.u.)")
# plt.title(f"Two Airy Resonances with Brillouin Side Peaks\nFSR = {FSR_GHz} GHz, finesse = {finesse}")
# plt.legend()
plt.tight_layout()
plt.show()
