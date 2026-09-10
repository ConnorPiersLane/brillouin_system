"""The measured instrument kernel of one peak, on the fine grid.

The kernel is the line profile stacked from calibration frames at the
detector position where a sample line sits (spectrum_fitting/epsf.py builds
it, per node along each line's track). It is the instrument response with
no functional form: VIPA line, camera blur, readout tails and the pixel
box, all measured. dho.dho_profile convolves the DHO core with it.

History: until 2026-09-10 a parametric Lorentzian x Gauss x tail x pixel
kernel with frozen per-peak constants lived next to it. Its Stokes wing
was ~30 % too heavy (2026-09-07) and its centre convention had to be
guarded against the plain Lorentzian's; the measured profile replaced it
everywhere and the parametric chain was removed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Step of the fine grid every kernel and profile lives on [px].
DX = 0.02
# Half-width of the stored kernel [px]. The far wing beyond this is < 1 %
# of the peak for the measured line and is absorbed by the fitted offset.
KERNEL_HALF_PX = 6.0
# Calibration frames whose line lies within this distance of the requested
# position contribute. Inside it the instrument HWHM changes by < 0.04 px
# while the sweep's ~0.3 px steps still cover every sub-pixel phase.
WINDOW_PX = 2.5
# Bandwidth of the local-quadratic smoother [px]: a few samples per node at
# the 41-point sweep density, well below the 0.4 px instrument half width.
SMOOTH_H_PX = 0.12


@dataclass(frozen=True)
class MeasuredKernel:
    """One peak's measured instrument response on the fine grid.

    u            offsets from the line centre [px], uniform step DX
    k            unit-area response (k.sum() * DX == 1)
    position_px  detector position the profile was built for
    n_frames     calibration frames that contributed
    g_median_px  the profile's HWHM [px] (diagnostic)
    """
    u: np.ndarray
    k: np.ndarray
    position_px: float
    n_frames: int
    g_median_px: float

    @property
    def x0(self) -> float:
        """Coordinate of k[0] relative to the line centre."""
        return float(self.u[0])


def _local_quadratic(u_samples, y_samples, grid, h):
    """Weighted local-quadratic regression evaluated at each grid node."""
    out = np.zeros_like(grid)
    for j, u0 in enumerate(grid):
        d = u_samples - u0
        w = np.exp(-0.5 * (d / h) ** 2)
        sel = w > 1e-3
        if sel.sum() < 4:
            out[j] = 0.0
            continue
        A = np.column_stack([np.ones(sel.sum()), d[sel], d[sel] ** 2])
        coef, *_ = np.linalg.lstsq(A * w[sel][:, None], y_samples[sel] * w[sel],
                                   rcond=None)
        out[j] = coef[0]
    return out


def sample_peak_positions(fitter, frame, n_peaks: int = 2):
    """Sample-peak centres of a frame from a PLAIN Lorentzian fit — only a
    position is needed here (good to ~0.05 px; the nodes are 1 px apart and
    blended), so no instrument kernel enters. Returns (left, right) for two
    peaks and (outer_left, left, right, outer_right) for four."""
    saved_sample = fitter.sample_config
    try:
        fitter.update_sample_config(replace(saved_sample, fitting_model="lorentzian"))
        px, sline = fitter.get_px_sline_from_image(np.asarray(frame, dtype=float))
        r = fitter.fit(np.asarray(px, dtype=float), np.asarray(sline, dtype=float),
                       is_reference_mode=False, n_peaks=n_peaks)
    finally:
        fitter.update_sample_config(saved_sample)
    if not r.is_success:
        raise ValueError("Could not locate the sample peaks to build the "
                         "measured kernels.")
    if n_peaks == 4:
        return (float(r.outer_left_peak_center_px), float(r.left_peak_center_px),
                float(r.right_peak_center_px), float(r.outer_right_peak_center_px))
    return float(r.left_peak_center_px), float(r.right_peak_center_px)
