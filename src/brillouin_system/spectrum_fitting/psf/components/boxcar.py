"""Row-tilt boxcar — the smear of summing a tilted line over rows.

Single responsibility: the centred top-hat of full width `width` [px].
The spectral line is tilted on the sensor (measured per peak from
per-row positions, e.g. 0.147 px/row on the outer anti-Stokes order),
so the row-sum superimposes n_rows displaced copies — a top-hat of
width tilt * n_rows. The width is a MEASURED geometric constant, not a
fit parameter; it is carried only where the tilt is large enough to
matter (outer_left, whose measured detected profile is trapezoidal
with a ~1 px flat top, 2026-09-04). width <= 0 returns the identity.
Same (x0, k) convention as gaussian_kernel.
"""
import numpy as np


def boxcar_kernel(width: float, dx: float):
    if width <= 0:
        return 0.0, np.array([1.0])
    n = max(int(round(width / dx)), 1)
    return -0.5 * width, np.ones(n + 1)
