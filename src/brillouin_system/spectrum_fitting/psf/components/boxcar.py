"""Row-tilt boxcar — the smear of summing a tilted line over rows.

Single responsibility: the centred top-hat of full width `width` [px].
Every spectral line is tilted against the CCD columns by ONE physical
constant — a ~27 MHz/row frequency shear (a VIPA property) divided by
the track's local dispersion, measured stable across sweeps and a
realignment (Data/2026-9-2/tilt_all_runs.py) — so the 13-row sum
superimposes displaced copies: a top-hat of width tilt * n_rows. The
width is a MEASURED geometric constant, never fitted. In PRODUCTION
(2026-09-05) only the OUTER orders carry it (1.95 / 0.85 px): the
inner peaks' bell-weighted smear is Gauss+tail-shaped and lives in
their sigma/tau (a top-hat there is measurably the wrong shape).
width <= 0 returns the identity. Same (x0, k) convention as
gaussian_kernel.
"""
import numpy as np


def boxcar_kernel(width: float, dx: float):
    if width <= 0:
        return 0.0, np.array([1.0])
    n = max(int(round(width / dx)), 1)
    return -0.5 * width, np.ones(n + 1)
