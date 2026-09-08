"""Measured instrument kernel (ePSF) for the 'dho_x_psf' sample model.

The parametric kernel Lorentzian(g_inst) (x) Gauss(sigma) (x) ExpTail(tau)
(x) pixel was chosen for and validated on line CENTRES (the once-per-pixel
calibration sine). Its WINGS were never checked: stacking the elastic
calibration lines at sub-pixel resolution (2026-09-07, three fine sweeps
and every per-scan calibration of the day) shows the Stokes (right) kernel
carrying ~30 % too much intensity beyond 2.5 px. A DHO fitted with that
kernel attributes too much of the observed wing to the instrument and
reads the acoustic width 3-4 % too narrow, right peak more than left.
A symmetric wing error moves no centre, so the sine is blind to it.

This module builds the instrument response EMPIRICALLY, at the detector
position where a sample line sits, from the scan's own calibration frames:

  1. every calibration frame is fitted in reference mode (the production
     Lorentzian_x_psf chain) for its two centres and amplitudes;
  2. frames whose line falls within +-WINDOW_PX of the requested position
     are kept (the instrument width varies < 0.04 px inside that window;
     a 41-point sweep gives ~14-18 frames per peak, every sub-pixel phase);
  3. per frame: the OTHER inner peak is subtracted with its parametric
     model, the frame floor is taken from the peak-free ends of the axis
     (a baseline zone near the line would subtract the Lorentzian wing
     itself — measured 20-30 % wing loss at 4-5 px), the counts are divided
     by the fitted amplitude and re-indexed by u = px - centre;
  4. the pooled (u, y) samples are smoothed with a local-QUADRATIC kernel
     regression on the fine grid (a local mean broadens by h^2 K''/2 and
     bends the centres; local-quadratic does not — measured 2026-08-11);
  5. the curve is clipped at zero and normalised to unit area.

The result is the pixel-integrated instrument response measured on the
day, at that position, with no assumed functional form (the "effective
PSF" of star photometry). Validated on water 22-46 C: fit chi2 3x lower,
left and right acoustic widths equal, resonance unchanged to 0.01 MHz, a
single 41-point calibration reproduces the 401-point fine-sweep profile.

The kernel is centred on the reference fit's centre convention — the same
convention the calibration polynomials map to frequency — so a DHO
resonance fitted through it lands on the same axis as before. It carries
the instrument Lorentzian at that position, so the fitted DHO width is the
acoustic width, exactly as with the parametric kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from brillouin_system.spectrum_fitting.psf import DX, psf_profile

# Half-width of the stored kernel [px]. The far wing beyond this is < 1 %
# of the peak for the measured line (it falls faster than the parametric
# Lorentzian) and is absorbed by the fitted offset.
KERNEL_HALF_PX = 6.0
# Calibration frames whose line lies within this distance of the requested
# position contribute. Inside it the instrument HWHM changes by < 0.04 px
# while the sweep's ~0.3 px steps still cover every sub-pixel phase.
WINDOW_PX = 2.5
# Bandwidth of the local-quadratic smoother [px]: a few samples per node at
# the 41-point sweep density, well below the 0.4 px instrument half width.
SMOOTH_H_PX = 0.12
# Fraction of the pixel axis at each end taken as the peak-free floor zone.
FLOOR_FRACTION = 0.1
# Minimum number of contributing frames for a trustworthy profile.
MIN_FRAMES = 8


@dataclass(frozen=True)
class MeasuredKernel:
    """One peak's measured instrument response on the fine grid.

    u          offsets from the line centre [px], uniform step DX
    k          unit-area response (k.sum() * DX == 1)
    position_px  detector position the profile was built for
    n_frames   calibration frames that contributed
    g_median_px  median reference-fit Lorentzian HWHM [px] of those frames
               (diagnostic; the parametric chain's g_inst at this position)
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


def _reference_fits(calibration_data, fitter):
    """(px, sline, (amp, cen, width) left, (amp, cen, width) right) per frame."""
    out = []
    for block in calibration_data.measured_freqs:
        for point in block.cali_meas_points:
            px, sline = fitter.get_px_sline_from_image(
                np.asarray(point.frame, dtype=float))
            px = np.asarray(px, dtype=float)
            sline = np.asarray(sline, dtype=float)
            r = fitter.fit(px, sline, is_reference_mode=True, n_peaks=2)
            if not r.is_success:
                continue
            out.append((px, sline,
                        (float(r.left_peak_amplitude),
                         float(r.left_peak_center_px),
                         float(r.left_peak_width_px)),
                        (float(r.right_peak_amplitude),
                         float(r.right_peak_center_px),
                         float(r.right_peak_width_px))))
    return out


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


def build_measured_kernel(calibration_data, fitter, position_px: float,
                          peak_index: int,
                          reference_fits=None) -> MeasuredKernel:
    """The measured instrument response of inner peak `peak_index`
    (0 = left/anti-Stokes, 1 = right/Stokes) at `position_px`.

    reference_fits: the result of a previous _reference_fits call (the
    calibration is fitted once and reused for both peaks).
    """
    fits = reference_fits if reference_fits is not None else _reference_fits(
        calibration_data, fitter)
    sl = fitter.sline_config
    sig = (float(sl.psf_sigma_left_px), float(sl.psf_sigma_right_px))
    tau = (float(sl.psf_tau_left_px), float(sl.psf_tau_right_px))
    other = 1 - peak_index

    U, Y, G = [], [], []
    for px, sline, left, right in fits:
        amp, cen, wid = (left, right)[peak_index]
        if abs(cen - position_px) > WINDOW_PX:
            continue
        o_amp, o_cen, o_wid = (left, right)[other]
        y = sline - psf_profile(px, o_amp, o_cen, o_wid, sig[other], tau[other])
        n_floor = max(int(round(FLOOR_FRACTION * len(px))), 3)
        floor_zone = np.r_[y[:n_floor], y[-n_floor:]]
        y = (y - float(np.median(floor_zone))) / amp
        u = px - cen
        keep = np.abs(u) <= KERNEL_HALF_PX + 0.5
        U.extend(u[keep].tolist())
        Y.extend(y[keep].tolist())
        G.append(wid)

    if len(G) < MIN_FRAMES:
        raise ValueError(
            f"Only {len(G)} calibration frames fall within +-{WINDOW_PX} px "
            f"of px {position_px:.1f} for peak {peak_index}; a measured "
            f"kernel needs at least {MIN_FRAMES}. Is the sample line inside "
            f"the calibrated sweep?"
        )

    grid = np.arange(-KERNEL_HALF_PX, KERNEL_HALF_PX + DX / 2, DX)
    k = _local_quadratic(np.asarray(U), np.asarray(Y), grid, SMOOTH_H_PX)
    k = np.clip(k, 0.0, None)
    area = float(k.sum() * DX)
    if not area > 0.0:
        raise ValueError("Measured kernel has no positive area.")
    k = k / area
    return MeasuredKernel(u=grid, k=k, position_px=float(position_px),
                          n_frames=len(G), g_median_px=float(np.median(G)))


def sample_peak_positions(fitter, frame, n_peaks: int = 2):
    """Sample-peak centres of a frame from a PLAIN Lorentzian fit — only a
    position is needed here (good to ~0.05 px; the nodes are 1 px apart and
    blended), so no instrument kernel and no camera-PSF constants enter.
    Returns (left, right) for two peaks and (outer_left, left, right,
    outer_right) for four."""
    saved_sample, saved_ref = fitter.sample_config, fitter.reference_config
    # the reference model is swapped alongside only to pass the model-mixing
    # guard (that guard protects fitted SHIFTS; this fit yields a position
    # for a 1-px node ladder, where the 0.27 px convention offset is moot)
    try:
        fitter.update_sample_config(replace(saved_sample, fitting_model="lorentzian"))
        fitter.update_reference_config(replace(saved_ref, fitting_model="lorentzian"))
        px, sline = fitter.get_px_sline_from_image(np.asarray(frame, dtype=float))
        r = fitter.fit(np.asarray(px, dtype=float), np.asarray(sline, dtype=float),
                       is_reference_mode=False, n_peaks=n_peaks)
    finally:
        fitter.update_sample_config(saved_sample)
        fitter.update_reference_config(saved_ref)
    if not r.is_success:
        raise ValueError("Could not locate the sample peaks to build the "
                         "measured kernels.")
    if n_peaks == 4:
        return (float(r.outer_left_peak_center_px), float(r.left_peak_center_px),
                float(r.right_peak_center_px), float(r.outer_right_peak_center_px))
    return float(r.left_peak_center_px), float(r.right_peak_center_px)


def measured_kernels_for_frame(calibration_data, fitter, frame
                               ) -> tuple[MeasuredKernel, MeasuredKernel]:
    """Both inner-peak kernels for a scan: positions from `frame`, profiles
    from the scan's own calibration (fitted once)."""
    left_px, right_px = sample_peak_positions(fitter, frame)
    fits = _reference_fits(calibration_data, fitter)
    return (build_measured_kernel(calibration_data, fitter, left_px, 0, fits),
            build_measured_kernel(calibration_data, fitter, right_px, 1, fits))
