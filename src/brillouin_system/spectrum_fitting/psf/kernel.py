"""Composition of the detection kernel: Gauss ⊗ Tail ⊗ Pixel ⊗ Boxcar.

Single responsibility: combine the single-responsibility components
into ONE cached kernel on the fine grid. The constants (sigma, tau,
box) are FROZEN instrument constants supplied by the fitting config —
never fitted per frame. Coordinates are tracked explicitly: convolving
arrays sampled from a0 and b0 yields a grid starting at a0 + b0, so
the pixel top-hat (spanning -0.5..+0.5) and the one-sided tail
(starting at 0) end up correctly placed instead of silently shifting
the profile. The result is normalised to unit AREA (k.sum()*dx == 1).
"""
from functools import lru_cache

import numpy as np

from .components import (boxcar_kernel, gaussian_kernel, pixel_kernel,
                         tail_kernel)

# Fine grid step [px] for the convolution. The frozen constants were
# measured with 0.02 px; kernel features are ~0.25 px wide, so this is
# well sampled.
DX = 0.02
# Grid padding beyond the evaluated pixels so the tail is not truncated.
PAD_PX = 4.0


@lru_cache(maxsize=64)
def detection_kernel(sigma: float, tau: float, dx: float,
                     box: float = 0.0):
    """(x0, k): the full detection kernel for one peak."""
    g_x0, g = gaussian_kernel(sigma, dx)
    t_x0, t = tail_kernel(tau, dx)
    b_x0, b = pixel_kernel(dx)
    w_x0, w = boxcar_kernel(box, dx)

    k = np.convolve(np.convolve(np.convolve(g, t), b), w)
    k = k / (k.sum() * dx)
    return g_x0 + t_x0 + b_x0 + w_x0, k
