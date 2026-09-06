"""Measured row-tilt smear — the tilted line summed over the row band.

Single responsibility: the EXACT smear kernel of the 13-row sum. Each
spectral line is tilted against the CCD columns (one ~27 MHz/row
frequency shear / the track's local dispersion), so the row-sum
superimposes 13 displaced copies of the line, WEIGHTED by the measured
row-intensity profile. That profile (measured 2026-09-04 on the 9-2
sweeps, identical for all four peaks within noise: bell-shaped,
sd ~2.0 rows, with a slow decay toward high rows) is what makes the
smear neither a pure top-hat (that ignores the weighting — right span,
wrong shape: fixes widths, breaks centres) nor a pure Gauss+tail (right
shape class, wrong span at large tilts): it is BOTH, with no free
parameters.

The kernel is a comb of 13 weighted deltas at offsets
tilt * (row - centroid), with tilt = span/13 — `span` is the same
measured constant the top-hat used (tilt x 13 rows: 1.95 px outer_left,
0.85 px outer_right). Centred on the intensity centroid so the smear
itself displaces nothing; the skewed weighting still carries the
physical asymmetry. span <= 0 returns the identity. Same (x0, k)
convention as gaussian_kernel.

Production carries this for the OUTER orders only: the inner peaks'
smear is small enough that their fitted sigma/tau absorb it (and a
separately-modelled smear there was measured to do harm).
"""
import numpy as np

# Measured row-intensity weights over the 13-row band (rows 7..19),
# normalised to max 1 — Data/2026-9-2, 09-04 measurement (per-peak
# profiles identical within noise; this is the outer_left/left average
# shape, representative of all four).
ROW_WEIGHTS = np.array([0.02, 0.01, 0.08, 0.37, 0.79, 1.00, 0.86,
                        0.58, 0.36, 0.21, 0.13, 0.08, 0.05])


def row_smear_kernel(span: float, dx: float):
    """(x0, k): the measured row-profile smear of full span `span` [px]."""
    if span <= 0:
        return 0.0, np.array([1.0])
    n_rows = ROW_WEIGHTS.size
    tilt = span / n_rows                     # px per row
    w = ROW_WEIGHTS / ROW_WEIGHTS.sum()
    centroid = float(np.sum(np.arange(n_rows) * w))
    offsets = tilt * (np.arange(n_rows) - centroid)

    lo = float(offsets.min())
    n = int(round((offsets.max() - lo) / dx)) + 1
    k = np.zeros(n)
    for off, wt in zip(offsets, w):
        k[int(round((off - lo) / dx))] += wt
    return lo, k
