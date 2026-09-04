# -----------------------
# Helper functions
# -----------------------
import numpy as np
from math import erf, sqrt, pi

def sinc_cyc(ff):
    """sinc in cycles/pixel convention: sin(pi f)/(pi f)."""
    out = np.ones_like(ff, dtype=float)
    nz = np.abs(ff) > 1e-14
    out[nz] = np.sin(np.pi * ff[nz]) / (np.pi * ff[nz])
    return out


def gaussian(x, sigma, center):
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x - center)**2 / (2*sigma**2))

def pixel_integrated_gaussian_continuous(x, sigma, center):
    """
    Continuous pixel-integrated Gaussian:
        h(x) = integral_{x-1/2}^{x+1/2} g(u-center) du
    This equals (g * rect)(x) for unit-width rect.
    """
    x = np.asarray(x, dtype=float)
    a = (x + 0.5 - center) / (np.sqrt(2) * sigma)
    b = (x - 0.5 - center) / (np.sqrt(2) * sigma)
    erf_vec = np.vectorize(erf)
    return 0.5 * (erf_vec(a) - erf_vec(b))

def pixel_samples_gauss(n, sigma, center):
    """
    Exact pixel values at integer pixel centers n:
        s[n] = integral_{n-1/2}^{n+1/2} g(x-center) dx
    """
    n = np.asarray(n, dtype=float)
    a = (n + 0.5 - center) / (np.sqrt(2) * sigma)
    b = (n - 0.5 - center) / (np.sqrt(2) * sigma)
    erf_vec = np.vectorize(erf)
    return 0.5 * (erf_vec(a) - erf_vec(b))



def rect_func(x, width=1.0):
    return np.where(np.abs(x) <= width/2, 1.0, 0.0)

def lorentzian(x, gamma, center):
    """
    Lorentzian (area normalized).

    gamma = HWHM (half-width at half-maximum)
    """
    return (gamma / np.pi) / ((x - center)**2 + gamma**2)

def pixel_integrated_lorentzian_continuous(x, gamma, center):
    """
    Continuous pixel-integrated Lorentzian:
        h(x) = ∫_{x-1/2}^{x+1/2} L(u-center) du
    """
    x = np.asarray(x, dtype=float)
    a = (x + 0.5 - center) / gamma
    b = (x - 0.5 - center) / gamma
    return (1/np.pi) * (np.arctan(a) - np.arctan(b))

def pixel_samples_lorentzian(n, gamma, center):
    """
    Exact pixel values at integer pixel centers:
        s[n] = ∫_{n-1/2}^{n+1/2} L(x-center) dx
    """
    n = np.asarray(n, dtype=float)
    a = (n + 0.5 - center) / gamma
    b = (n - 0.5 - center) / gamma
    return (1/np.pi) * (np.arctan(a) - np.arctan(b))

def sinc_reconstruct(x, n, s):
    """
    Reconstruct continuous signal from discrete samples:
        h_rec(x) = sum_n s[n] sinc(x-n)
    """
    x = np.asarray(x, dtype=float)
    h_rec = np.zeros_like(x)
    for ni, si in zip(n, s):
        h_rec += si * sinc_cyc(x - ni)
    return h_rec

def estimate_center_from_peak(x, y):
    """
    Center estimate = x position of the maximum.
    """
    return x[np.argmax(y)]

def estimate_center_from_centroid(x, y):
    """
    Center estimate = first moment.
    Assumes y >= 0 and localized.
    """
    area = np.trapezoid(y, x)
    return np.trapezoid(x * y, x) / area