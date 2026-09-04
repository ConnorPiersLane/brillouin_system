"""The single-responsibility building blocks of the detection model.

Each module holds exactly one function:

    lorentzian.py        the VIPA line itself (the physics)
    gaussian.py          symmetric optical/charge-diffusion blur
    exponential_tail.py  one-sided readout/aberration tail
    boxcar.py            measured row-tilt smear of the row-sum
    pixel.py             1 px camera sampling aperture

Kernel components share one convention: (x0, k) with x0 the coordinate
of k[0] relative to the centre, k unnormalised; identity for disabled
components. kernel.py composes them; nothing here is composed twice.
"""
from .boxcar import boxcar_kernel
from .exponential_tail import tail_kernel
from .gaussian import gaussian_kernel
from .lorentzian import lorentzian
from .pixel import pixel_kernel

__all__ = ["boxcar_kernel", "tail_kernel", "gaussian_kernel",
           "lorentzian", "pixel_kernel"]
