"""Frequency tracks for the OUTER VIPA orders — the opt-in 4-peak chain.

ANALYSIS-SIDE ONLY. Production calibration is and stays two-track
(calibration.calibrate + CalibrationCalculator); nothing in the standard
chain imports this module. It exists for the analyses that fit all four
VIPA orders (SpectrumFitter.fit(..., n_peaks=4)) and need to convert the
outer peaks' pixel positions to frequency — e.g. the four-peak precision
panel, where each order's own local dispersion turns a pixel scatter into
MHz.

It works because the same stored calibration frames that produce the
production tracks carry the EOM sidebands in the outer orders too, when
the ROI was wide enough to contain them (the 4-peak-ROI sessions, e.g.
8-13/8-14 — the same frames the outer taus were measured from). build()
re-runs the one fitting pass over those frames with n_peaks=4 and tracks
the outer sideband positions against the set microwave frequency, exactly
as calibrate() does for the main pair. The production calibration of the
scan is not touched or recomputed.

Precondition guard: on a main-pair-only ROI there are no outer peaks to
track, so build() raises instead of fitting garbage.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationData,
    sort_xy,
)
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


@dataclass
class OuterOrderTracks:
    """px -> GHz maps of the two outer VIPA orders, mirroring the production
    CalibrationCalculator interface (freq_* / dfreq_dpx_* / df_*)."""
    degree: int = 1
    freq_outer_left: Optional[np.ndarray] = field(default=None)
    freq_outer_right: Optional[np.ndarray] = field(default=None)

    # The measured sideband points behind the polynomials (sorted by px) —
    # for plots and residual diagnostics, never used to evaluate the track.
    outer_left_px_points: Optional[np.ndarray] = field(default=None)
    outer_left_freq_points: Optional[np.ndarray] = field(default=None)
    outer_right_px_points: Optional[np.ndarray] = field(default=None)
    outer_right_freq_points: Optional[np.ndarray] = field(default=None)

    def freq_outer_left_ghz(self, px):
        """Frequency of the outer-left peak [GHz] at pixel position px."""
        return np.polyval(self.freq_outer_left, px)

    def freq_outer_right_ghz(self, px):
        """Frequency of the outer-right peak [GHz] at pixel position px."""
        return np.polyval(self.freq_outer_right, px)

    def dfreq_dpx_outer_left(self, px):
        """Local dispersion d(freq)/d(px) of the outer-left track [GHz/px]."""
        return np.polyval(np.polyder(self.freq_outer_left, m=1), px)

    def dfreq_dpx_outer_right(self, px):
        """Local dispersion d(freq)/d(px) of the outer-right track [GHz/px]."""
        return np.polyval(np.polyder(self.freq_outer_right, m=1), px)

    def df_outer_left(self, px, dpx):
        """Convert dpx to GHz using the outer-left track's local slope."""
        return self.dfreq_dpx_outer_left(px) * dpx

    def df_outer_right(self, px, dpx):
        """Convert dpx to GHz using the outer-right track's local slope."""
        return self.dfreq_dpx_outer_right(px) * dpx


def build_outer_order_tracks(data: CalibrationData, polyfit_degree: int,
                             fitter: SpectrumFitter | None = None,
                             min_fits: int = 5) -> OuterOrderTracks:
    """Fit the outer-order tracks from a calibration's raw frames.

    Pass the same fitter used for the samples when working on a scan's own
    calibration — it carries that scan's frozen row band (a one-row
    difference biases the peaks by ~3-4 MHz), same rule as calibrate().

    Raises when fewer than min_fits frames yield a successful 4-peak fit:
    that means the calibration ROI does not contain the outer orders (or
    the reference thresholds miss them) and there is nothing to track.
    """
    sf = fitter if fitter is not None else SpectrumFitter()

    outer_left_px, outer_right_px, freqs = [], [], []
    n_frames = 0
    for freq_block in data.measured_freqs:
        for point in freq_block.cali_meas_points:
            n_frames += 1
            px, sline = sf.get_px_sline_from_image(point.frame)
            fs = sf.fit(px, sline, is_reference_mode=True, n_peaks=4)
            if fs.is_success and fs.outer_left_peak_center_px is not None:
                outer_left_px.append(fs.outer_left_peak_center_px)
                outer_right_px.append(fs.outer_right_peak_center_px)
                freqs.append(point.microwave_freq)

    if len(freqs) < min_fits:
        raise ValueError(
            f"Only {len(freqs)} of {n_frames} calibration frames produced a "
            f"successful 4-peak fit (need >= {min_fits}). The outer-order "
            f"tracks need calibration frames whose ROI contains all four "
            f"VIPA orders (a 4-peak-ROI session, e.g. 8-13/8-14) — a "
            f"main-pair-only calibration cannot supply them."
        )

    freqs = np.asarray(freqs, dtype=float)
    outer_left_px = np.asarray(outer_left_px, dtype=float)
    outer_right_px = np.asarray(outer_right_px, dtype=float)

    ol_px_sorted, ol_freq_sorted = sort_xy(outer_left_px, freqs)
    or_px_sorted, or_freq_sorted = sort_xy(outer_right_px, freqs)

    return OuterOrderTracks(
        degree=polyfit_degree,
        freq_outer_left=np.polyfit(outer_left_px, freqs, polyfit_degree),
        freq_outer_right=np.polyfit(outer_right_px, freqs, polyfit_degree),
        outer_left_px_points=ol_px_sorted,
        outer_left_freq_points=ol_freq_sorted,
        outer_right_px_points=or_px_sorted,
        outer_right_freq_points=or_freq_sorted,
    )


@dataclass
class FourPeakShift:
    """The four per-order frequency estimates of one fit and their
    inverse-variance combination. Frequencies in GHz, ordered left to
    right on the detector: outer_left, left, right, outer_right."""
    freqs_ghz: tuple[float, float, float, float]
    weights: tuple[float, float, float, float]
    combined_ghz: float


def four_peak_shift(fs: FittedSpectrum,
                    calc: CalibrationCalculator,
                    tracks: OuterOrderTracks) -> FourPeakShift:
    """ONE frequency measurement from the position estimates of all four peaks.

    Each order's fitted centre maps to the Brillouin shift through its own
    track (the two production tracks for the inner pair, the outer tracks
    from build_outer_order_tracks), giving four estimates of the same
    quantity; they are combined by inverse-variance weighting.

    The weights are the Thompson photon terms, which only need RELATIVE
    variances, so the gain and all shared constants cancel:

        var_i  ∝  s_i^2 / N_i  ∝  (w_i a_i)^2 / (amp_i w_i)  =  a_i^2 w_i / amp_i

    with w the fitted width [px], a the track's local dispersion [GHz/px]
    and amp the fitted amplitude (N ∝ amp*w, the exact peak area). The
    photon term dominates the per-peak budget, so richer weights (read
    noise, pedestal) would move the combination negligibly while dragging
    in the camera gain.
    """
    if not fs.is_success or fs.outer_left_peak_center_px is None:
        raise ValueError(
            "four_peak_shift needs a successful 4-peak fit — run "
            "SpectrumFitter.fit(..., n_peaks=4) on a spectrum whose ROI "
            "contains the outer VIPA orders."
        )

    peaks = [
        (fs.outer_left_peak_center_px, fs.outer_left_peak_width_px,
         fs.outer_left_peak_amplitude,
         tracks.freq_outer_left_ghz, tracks.dfreq_dpx_outer_left),
        (fs.left_peak_center_px, fs.left_peak_width_px,
         fs.left_peak_amplitude,
         calc.freq_left_peak, calc.dfreq_dpx_left_peak),
        (fs.right_peak_center_px, fs.right_peak_width_px,
         fs.right_peak_amplitude,
         calc.freq_right_peak, calc.dfreq_dpx_right_peak),
        (fs.outer_right_peak_center_px, fs.outer_right_peak_width_px,
         fs.outer_right_peak_amplitude,
         tracks.freq_outer_right_ghz, tracks.dfreq_dpx_outer_right),
    ]

    freqs, weights = [], []
    for cen, wid, amp, freq_of_px, slope_of_px in peaks:
        freqs.append(float(freq_of_px(cen)))
        a = float(slope_of_px(cen))
        var = a * a * float(wid) / max(float(amp), 1e-12)
        weights.append(1.0 / var)

    w = np.asarray(weights, dtype=float)
    f = np.asarray(freqs, dtype=float)
    combined = float(np.sum(w * f) / np.sum(w))

    return FourPeakShift(
        freqs_ghz=tuple(f.tolist()),
        weights=tuple((w / np.sum(w)).tolist()),
        combined_ghz=combined,
    )
