"""The detected profile of one Lorentzian line.

Single responsibility: evaluate Lorentzian(gamma) ⊗ detection kernel
at the requested pixels — the workhorse every peak model is built
from. The fitted parameters are amp/cen/gamma; sigma/tau/box are the
frozen instrument constants (see the fitting config).
"""
import numpy as np

from .components import lorentzian
from .kernel import DX, PAD_PX, detection_kernel


def psf_profile(px, amp, cen, gamma, sigma, tau, box=0.0):
    """Lorentzian(gamma) through the pixel response, evaluated at px.

    amp is the peak height of the underlying Lorentzian, matching the
    plain pixel-integrated Lorentzian model's convention. box: measured
    row-tilt boxcar full width [px], 0 to disable.
    """
    px = np.asarray(px, dtype=float)

    lo = float(px.min()) - PAD_PX
    hi = float(px.max()) + PAD_PX
    n = int(round((hi - lo) / DX)) + 1
    xf = lo + DX * np.arange(n)

    lor = lorentzian(xf, cen, gamma)

    k_x0, k = detection_kernel(float(sigma), float(tau), DX, float(box))
    conv = np.convolve(lor, k) * DX
    conv_x0 = xf[0] + k_x0
    conv_x = conv_x0 + DX * np.arange(conv.size)

    return float(amp) * np.interp(px, conv_x, conv)
